#!/usr/bin/env python3
"""
Batch Inference for TableFormerV2.
Environment: tableformer
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
from docling_ibm_models.tableformer_v2 import TableFormerV2
from tqdm import tqdm

sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts")
from otsl_utils import otsl_to_html

ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

def run_batch(data_root: Path, output_dir: Path, device: str = "cuda:0", model_id: str = "docling-project/TableFormerV2", languages=None):
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TableFormerV2] Loading model on {device}...")
    dev = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = TableFormerV2.from_pretrained(model_id).to(dev)
    model.eval()

    transform = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    target_languages = languages or ALL_LANGUAGES
    for lang in target_languages:
        ann_file = data_root / "annotations" / f"{lang}.jsonl"

        with open(ann_file, "r", encoding="utf-8") as f:

            items = [json.loads(line) for line in f if line.strip()]


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
            print(f"[TableFormerV2] {lang} already completed ({len(processed_ids)} records). Skipping.")
            continue

        records = []
        for item in tqdm(items, desc=f"TableFormer [{lang}]"):
            if item["image_id"] in processed_ids:
                continue
            img_rel = item["image"]
            img_path = str(data_root / img_rel)
            img_id = item["image_id"]
            bucket = item.get("complexity_bucket", "unknown")

            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(dev)

                s = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(img_tensor, tokenizer=tokenizer, max_length=512)
                elapse = time.perf_counter() - s

                gen_tokens = tokenizer.decode(out["generated_ids"][0], skip_special_tokens=True)
                bboxes = out["predicted_bboxes"][0].cpu().numpy().tolist() if out["predicted_bboxes"] is not None else []
                html_text = otsl_to_html(gen_tokens)

                rec = {
                    "image_id": img_id,
                    "language": lang,
                    "complexity_bucket": bucket,
                    "prediction_html": html_text,
                    "cell_bboxes": bboxes,
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
            records.append(rec)
            with open(out_file, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[TableFormerV2] Finished {lang} ({len(records)} records).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--language", action="append", dest="languages", default=None)
    args = parser.parse_args()

    run_batch(args.data_root, args.output_dir, args.device, languages=args.languages)
