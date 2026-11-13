#!/usr/bin/env python3
"""
Batch Inference for SPRINT.
Environment: sprint
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts")
from otsl_utils import otsl_to_html

ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

def run_batch(data_root: Path, output_dir: Path, device: str = "cuda:0", languages=None):
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SPRINT] Loading model on {device}...")
    model_weight_path = "/home/tanish/scratch/tanishq/table-understanding/repos/SPRINT/weights/SPRINT_merged.pt"
    dev = torch.device(device)
    model = torch.load(model_weight_path, map_location=dev)
    model.to(dev)
    model.eval()

    test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

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

                print(f"[SPRINT] {lang} already completed ({len(existing)} records). Skipping.")

                continue

        records = []
        for item in tqdm(items, desc=f"SPRINT [{lang}]"):
            img_rel = item["image"]
            img_path = str(data_root / img_rel)
            img_id = item["image_id"]
            bucket = item.get("complexity_bucket", "unknown")

            try:
                img = Image.open(img_path)
                img_tensor = test_transforms(img).unsqueeze(0).to(dev)

                s = time.perf_counter()
                with torch.no_grad():
                    output = model(img_tensor, None, return_preds=True)
                elapse = time.perf_counter() - s

                otsl_pred = output["preds"][0][0]
                html_pred = otsl_to_html(otsl_pred)

                rec = {
                    "image_id": img_id,
                    "language": lang,
                    "complexity_bucket": bucket,
                    "prediction_html": html_pred,
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
            records.append(rec)

        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[SPRINT] Finished {lang} ({len(records)} records).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--language", action="append", dest="languages", default=None)
    args = parser.parse_args()

    run_batch(args.data_root, args.output_dir, args.device, languages=args.languages)
