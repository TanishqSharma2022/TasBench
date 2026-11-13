#!/usr/bin/env python3
"""
Batch Inference for SLANet-plus.
Environment: slanet
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

from rapid_table import RapidTable, RapidTableInput, ModelType

ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

def run_batch(data_root: Path, output_dir: Path, device: str = "cuda:0", languages=None):
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SLANet-plus] Initializing engine on {device}...")
    cfg = RapidTableInput(model_type=ModelType.SLANETPLUS, use_ocr=False)
    engine = RapidTable(cfg)

    target_languages = languages or ALL_LANGUAGES
    for lang in target_languages:
        ann_file = data_root / "annotations" / f"{lang}.jsonl"

        with open(ann_file, "r", encoding="utf-8") as f:

            items = [json.loads(line) for line in f if line.strip()]


        out_file = preds_dir / f"{lang}.jsonl"

        if out_file.exists():

            with open(out_file, "r") as f:

                existing = [line for line in f if line.strip()]

            if len(existing) >= len(items):

                print(f"[SLANet-plus] {lang} already completed ({len(existing)} records). Skipping.")

                continue

        records = []
        for item in tqdm(items, desc=f"SLANet [{lang}]"):
            img_rel = item["image"]
            img_path = str(data_root / img_rel)
            img_id = item["image_id"]
            bucket = item.get("complexity_bucket", "unknown")

            try:
                img = cv2.imread(img_path)
                s = time.perf_counter()
                pred_html_tokens, cell_bboxes = engine.table_structure([img])
                elapse = time.perf_counter() - s

                tokens = pred_html_tokens[0][0] if pred_html_tokens and len(pred_html_tokens[0]) > 0 else []
                bboxes = cell_bboxes[0].tolist() if len(cell_bboxes) > 0 and len(cell_bboxes[0]) > 0 else []
                html = "".join(tokens)
                if not html.startswith("<table>"):
                    html = f"<table>{html}</table>"

                rec = {
                    "image_id": img_id,
                    "language": lang,
                    "complexity_bucket": bucket,
                    "prediction_html": html,
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

        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[SLANet-plus] Finished {lang} ({len(records)} records).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--language", action="append", dest="languages", default=None)
    args = parser.parse_args()

    run_batch(args.data_root, args.output_dir, args.device, languages=args.languages)
