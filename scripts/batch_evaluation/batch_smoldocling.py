#!/usr/bin/env python3
"""
Batch Inference for SmolDocling-256M.
Environment: smoldocling
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm

sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts")
from otsl_utils import otsl_to_html

ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

def run_batch(data_root: Path, output_dir: Path, device: str = "cuda:0", languages=None, shard_id: int = 0, num_shards: int = 1):
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    target_languages = languages or ALL_LANGUAGES

    print(f"[SmolDocling-256M] Loading model on {device} (shard {shard_id}/{num_shards})...")
    model_id = "ds4sd/SmolDocling-256M-preview"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this table to docling."}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    for lang in target_languages:
        ann_file = data_root / "annotations" / f"{lang}.jsonl"
        with open(ann_file, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]

        if num_shards > 1:
            items = items[shard_id::num_shards]
            out_file = preds_dir / f"{lang}_shard_{shard_id}.jsonl"
        else:
            out_file = preds_dir / f"{lang}.jsonl"

        processed_ids = set()
        if out_file.exists():
            with open(out_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            processed_ids.add(json.loads(line)["image_id"])
                        except Exception:
                            pass

        if len(processed_ids) >= len(items):
            print(f"[SmolDocling] {lang} (shard {shard_id}) already completed ({len(processed_ids)} records). Skipping.")
            continue

        print(f"[SmolDocling] Processing {lang} (shard {shard_id}): {len(processed_ids)}/{len(items)} done.")

        with open(out_file, "a", encoding="utf-8") as f_out:
            for item in tqdm(items, desc=f"SmolDocling [{lang}-shard{shard_id}]"):
                img_id = item["image_id"]
                if img_id in processed_ids:
                    continue

                img_rel = item["image"]
                img_path = str(data_root / img_rel)
                bucket = item.get("complexity_bucket", "unknown")

                try:
                    image = Image.open(img_path).convert("RGB")
                    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

                    s = time.perf_counter()
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=1024)
                    elapse = time.perf_counter() - s

                    prompt_len = inputs.input_ids.shape[1]
                    trimmed_ids = generated_ids[:, prompt_len:]
                    doctags = processor.batch_decode(trimmed_ids, skip_special_tokens=False)[0].lstrip()
                    html_text = otsl_to_html(doctags)

                    rec = {
                        "image_id": img_id,
                        "language": lang,
                        "complexity_bucket": bucket,
                        "prediction_html": html_text,
                        "cell_bboxes": [],
                        "elapse_sec": elapse,
                        "status": "ok"
                    }
                except Exception as e:
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
        print(f"[SmolDocling] Finished {lang} ({len(records)} records).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--language", action="append", dest="languages", default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    run_batch(args.data_root, args.output_dir, args.device, args.languages, shard_id=args.shard_id, num_shards=args.num_shards)
