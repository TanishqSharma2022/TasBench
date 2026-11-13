#!/usr/bin/env python3
"""
Batch Inference for TATR v1.1-All.
Environment: tables-detr
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

repo_src = "/home/tanish/scratch/tanishq/table-understanding/repos/table-transformer/src"
sys.path.insert(0, repo_src)
os.chdir(repo_src)
from inference import TableExtractionPipeline

ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

def run_batch(data_root: Path, output_dir: Path, device: str = "cuda:0", languages=None):
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    config_path = "/home/tanish/scratch/tanishq/table-understanding/repos/table-transformer/src/structure_config.json.pre-potatr"
    model_path = "/home/tanish/scratch/tanishq/table-understanding/repos/table-transformer/weights/TATR-v1.1-All-msft.pth"

    print(f"[TATR] Loading model on {device}...")
    pipe = TableExtractionPipeline(
        str_config_path=config_path,
        str_model_path=model_path,
        str_device=device
    )

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

                print(f"[TATR] {lang} already completed ({len(existing)} records). Skipping.")

                continue

        records = []
        for item in tqdm(items, desc=f"TATR [{lang}]"):
            img_rel = item["image"]
            img_path = str(data_root / img_rel)
            img_id = item["image_id"]
            bucket = item.get("complexity_bucket", "unknown")

            try:
                img = Image.open(img_path).convert("RGB")
                s = time.perf_counter()
                extracted_tables = pipe.recognize(
                    img,
                    tokens=[],
                    out_html=True,
                    out_cells=True
                )
                elapse = time.perf_counter() - s

                html = extracted_tables.get("html", [""])[0] if extracted_tables.get("html") else ""
                cells_raw = extracted_tables.get("cells", [[]])[0] if extracted_tables.get("cells") else []
                cell_bboxes = []
                for c in cells_raw:
                    if isinstance(c, dict) and "bbox" in c:
                        cell_bboxes.append(c["bbox"])
                    elif isinstance(c, (list, tuple)):
                        cell_bboxes.append(c)

                rec = {
                    "image_id": img_id,
                    "language": lang,
                    "complexity_bucket": bucket,
                    "prediction_html": html,
                    "cell_bboxes": cell_bboxes,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--language", action="append", dest="languages", default=None)
    args = parser.parse_args()

    run_batch(args.data_root, args.output_dir, args.device, languages=args.languages)
