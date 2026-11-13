#!/usr/bin/env python3
"""
Batch Inference for PIX-TAB.
Environment: mtltabnet
"""
import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

REPO_DIR = "/home/tanish/scratch/tanishq/table-understanding/repos/PIX-TAB"
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

import mmocr.models
import mmocr.datasets
import mmocr.datasets.pipelines
from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference

from OTSL_parse import OTSLTableParser
from bbox import parse_papp_otsl_bboxes

ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

def run_batch(data_root: Path, output_dir: Path, device: str = "cuda:0", languages=None):
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PIX-TAB] Loading model on {device}...")
    config_file = f"{REPO_DIR}/full-pubtabnet-500k-resume-epoch18/table_master_ResnetExtract_Ranger_0705_cell150_batch4.py"
    checkpoint_file = "/home/tanish/scratch/tanishq/table-understanding/repos/PIX-TAB/PIX-TAB-organized/checkpoints/full-pubtabnet-500k-resume-epoch18/epoch_25.pth"

    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)
    if model.cfg.data.test.get("type") == "ConcatDataset":
        model.cfg.data.test.pipeline = model.cfg.data.test["datasets"][0].pipeline

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
            print(f"[PIX-TAB] {lang} already completed ({len(processed_ids)} records). Skipping.")
            continue

        records = []
        for item in tqdm(items, desc=f"PIX-TAB [{lang}]"):
            if item["image_id"] in processed_ids:
                continue
            img_rel = item["image"]
            img_path = str(data_root / img_rel)
            img_id = item["image_id"]
            bucket = item.get("complexity_bucket", "unknown")

            try:
                s = time.perf_counter()
                result = model_inference(model, img_path)
                elapse = time.perf_counter() - s

                orig_img = Image.open(img_path)
                orig_w, orig_h = orig_img.size
                tokens_str = result.get("text", "")

                parser = OTSLTableParser(tokens_str)
                parsed = parser.parse()
                clean_html = re.sub(r"Cell_\d+_\d+", "", parsed["html"])
                clean_html = re.sub(r' data-bbox=\"\[.*?\]\"', "", clean_html)
                clean_html = "".join(clean_html.split("\n"))
                if not clean_html.startswith("<table>"):
                    clean_html = f"<table>{clean_html}</table>"

                cell_bboxes = parse_papp_otsl_bboxes(tokens_str, (orig_w, orig_h), model_input_size=(480, 480))

                rec = {
                    "image_id": img_id,
                    "language": lang,
                    "complexity_bucket": bucket,
                    "prediction_html": clean_html,
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
            with open(out_file, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[PIX-TAB] Finished {lang} ({len(records)} records).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--language", action="append", dest="languages", default=None)
    args = parser.parse_args()

    run_batch(args.data_root, args.output_dir, args.device, languages=args.languages)
