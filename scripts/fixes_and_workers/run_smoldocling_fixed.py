#!/usr/bin/env python3
"""
Fixed & Accelerated Batch Inference for SmolDocling-256M.
Environment: smoldocling
Uses docling-core's official parse_otsl_table_content to parse DocTags tokens,
eliminating the character-level parsing bug that corrupted table cells with OCR text.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from docling_core.types.doc.utils import parse_otsl_table_content
from tqdm import tqdm

ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

def doctags_to_html(doctags_str: str) -> str:
    """Convert SmolDocling DocTags string to standardized HTML table."""
    dt = doctags_str.strip()
    if "<otsl>" in dt:
        otsl_chunk = dt[dt.find("<otsl>"):]
        if "</otsl>" in otsl_chunk:
            otsl_chunk = otsl_chunk[:otsl_chunk.find("</otsl>") + 7]
    else:
        otsl_chunk = dt

    try:
        td = parse_otsl_table_content(otsl_chunk)
        if td.num_rows == 0 or td.num_cols == 0 or not td.table_cells:
            return "<table><tbody></tbody></table>"

        rows = defaultdict(list)
        for c in td.table_cells:
            rows[c.start_row_offset_idx].append(c)

        html = ["<table><tbody>"]
        for r_idx in range(td.num_rows):
            html.append("<tr>")
            row_cells = sorted(rows[r_idx], key=lambda x: x.start_col_offset_idx)
            for c in row_cells:
                rs = f' rowspan="{c.row_span}"' if c.row_span > 1 else ""
                cs = f' colspan="{c.col_span}"' if c.col_span > 1 else ""
                html.append(f"<td{rs}{cs}></td>")
            html.append("</tr>")
        html.append("</tbody></table>")
        return "".join(html)
    except Exception:
        return "<table><tbody></tbody></table>"

def run_batch(data_root: Path, output_dir: Path, device: str = "cuda:3", languages=None, batch_size: int = 8):
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SmolDocling-256M] Loading model on {device}...")
    model_id = "ds4sd/SmolDocling-256M-preview"
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"

    dev = torch.device(device)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa"
    ).to(dev)
    model.eval()

    eos_token_ids = [2, 1, processor.tokenizer.convert_tokens_to_ids("<end_of_utterance>")]

    target_languages = languages or ALL_LANGUAGES
    for lang in target_languages:
        ann_file = data_root / "annotations" / f"{lang}.jsonl"
        with open(ann_file, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]

        out_file = preds_dir / f"{lang}.jsonl"
        
        # Check if already completed
        if out_file.exists():
            with open(out_file, "r", encoding="utf-8") as f:
                existing = [line for line in f if line.strip()]
            if len(existing) >= len(items):
                print(f"[SmolDocling] {lang} already completed ({len(existing)} records). Skipping.")
                continue

        print(f"[SmolDocling] Starting inference for {lang} ({len(items)} samples) on {device}...")

        records = []
        with open(out_file, "w", encoding="utf-8") as f_out:
            for b_idx in tqdm(range(0, len(items), batch_size), desc=f"SmolDocling [{lang}]"):
                batch_items = items[b_idx: b_idx + batch_size]
                
                batch_images = []
                batch_prompts = []
                for it in batch_items:
                    img_path = data_root / it["image"]
                    try:
                        img = Image.open(str(img_path)).convert("RGB")
                    except Exception:
                        img = Image.new("RGB", (224, 224), color="white")
                    batch_images.append(img)
                    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Convert this table to docling."}]}]
                    prompt = processor.apply_chat_template(msg, add_generation_prompt=True)
                    batch_prompts.append(prompt)

                try:
                    inputs = processor(text=batch_prompts, images=batch_images, return_tensors="pt", padding=True).to(dev)
                    s = time.perf_counter()
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            eos_token_id=eos_token_ids
                        )
                    elapse = (time.perf_counter() - s) / len(batch_items)

                    prompt_len = inputs.input_ids.shape[1]
                    trimmed_ids = generated_ids[:, prompt_len:]
                    decoded_texts = processor.batch_decode(trimmed_ids, skip_special_tokens=False)

                    for idx, it in enumerate(batch_items):
                        img_id = it["image_id"]
                        bucket = it.get("complexity_bucket", "unknown")
                        raw_dt = decoded_texts[idx]
                        html_pred = doctags_to_html(raw_dt)

                        rec = {
                            "image_id": img_id,
                            "language": lang,
                            "complexity_bucket": bucket,
                            "prediction_html": html_pred,
                            "cell_bboxes": [],
                            "elapse_sec": elapse,
                            "status": "ok"
                        }
                        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f_out.flush()
                        records.append(rec)

                except Exception as e:
                    for it in batch_items:
                        img_id = it["image_id"]
                        bucket = it.get("complexity_bucket", "unknown")
                        rec = {
                            "image_id": img_id,
                            "language": lang,
                            "complexity_bucket": bucket,
                            "prediction_html": "<table><tbody></tbody></table>",
                            "cell_bboxes": [],
                            "elapse_sec": 0.0,
                            "status": "error",
                            "error": str(e)
                        }
                        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f_out.flush()
                        records.append(rec)

        print(f"[SmolDocling] Finished {lang} ({len(records)} records).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--language", action="append", dest="languages", default=None)
    args = parser.parse_args()

    run_batch(
        args.data_root,
        args.output_dir,
        args.device,
        languages=args.languages,
        batch_size=args.batch_size
    )
