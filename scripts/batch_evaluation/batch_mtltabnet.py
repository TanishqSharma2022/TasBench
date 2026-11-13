#!/usr/bin/env python3
"""
Batch Inference for MTL-TabNet.
Environment: mtltabnet
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_DIR = "/home/tanish/scratch/tanishq/table-understanding/repos/MTL-TabNet"
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

import mmocr.models
import mmocr.datasets
import mmocr.datasets.pipelines
from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference

ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

def clean_mtltabnet_tokens(token_str):
    tokens = token_str.split(',')
    new_tokens = []
    p = 0
    while p < len(tokens):
        t = tokens[p]
        if t == '<td':
            if p + 4 < len(tokens) and (tokens[p+2].startswith(' colspan=') or tokens[p+2].startswith(' rowspan=')):
                new_tokens.append(''.join(tokens[p:p+5]))
                p += 5
            elif p + 3 < len(tokens) and (tokens[p+1].startswith(' colspan=') or tokens[p+1].startswith(' rowspan=')):
                new_tokens.append(''.join(tokens[p:p+4]))
                p += 4
            else:
                new_tokens.append(t)
                p += 1
        elif t.startswith('<eb') and t.endswith('</eb>'):
            new_tokens.append('<td></td>')
            p += 1
        else:
            new_tokens.append(t)
            p += 1
    html = ''.join(new_tokens)
    for i in range(1, 20):
        html = html.replace(f'<eb{i}></eb{i}>', '<td></td>')
    html = html.replace('<eb></eb>', '<td></td>')
    if not html.startswith('<table>'):
        html = f'<table>{html}'
    if not html.endswith('</table>'):
        html = f'{html}</table>'
    return html

def run_batch(data_root: Path, output_dir: Path, device: str = "cuda:0", languages=None):
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MTL-TabNet] Loading model on {device}...")
    config_file = f"{REPO_DIR}/configs/textrecog/master/table_master_ResnetExtract_Ranger_0705.py"
    checkpoint_file = f"{REPO_DIR}/table_recognition/models/PubTabNet/epoch_19.pth"

    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)
    if model.cfg.data.test.get("type") == "ConcatDataset":
        model.cfg.data.test.pipeline = model.cfg.data.test["datasets"][0].pipeline

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

                print(f"[MTL-TabNet] {lang} already completed ({len(existing)} records). Skipping.")

                continue

        records = []
        for item in tqdm(items, desc=f"MTL-TabNet [{lang}]"):
            img_rel = item["image"]
            img_path = str(data_root / img_rel)
            img_id = item["image_id"]
            bucket = item.get("complexity_bucket", "unknown")

            try:
                s = time.perf_counter()
                result = model_inference(model, img_path)
                elapse = time.perf_counter() - s

                tokens_str = result.get("text", "")
                raw_bboxes = np.array(result.get("bbox", []))
                html_text = clean_mtltabnet_tokens(tokens_str)

                cell_bboxes = []
                if len(raw_bboxes) > 0:
                    new_bboxes = np.empty_like(raw_bboxes)
                    new_bboxes[..., 0] = raw_bboxes[..., 0] - raw_bboxes[..., 2] / 2.0
                    new_bboxes[..., 1] = raw_bboxes[..., 1] - raw_bboxes[..., 3] / 2.0
                    new_bboxes[..., 2] = raw_bboxes[..., 0] + raw_bboxes[..., 2] / 2.0
                    new_bboxes[..., 3] = raw_bboxes[..., 1] + raw_bboxes[..., 3] / 2.0
                    for b in new_bboxes:
                        if b[2] > b[0] and b[3] > b[1]:
                            cell_bboxes.append([round(float(x), 2) for x in b])

                rec = {
                    "image_id": img_id,
                    "language": lang,
                    "complexity_bucket": bucket,
                    "prediction_html": html_text,
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
        print(f"[MTL-TabNet] Finished {lang} ({len(records)} records).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--language", action="append", dest="languages", default=None)
    args = parser.parse_args()

    run_batch(args.data_root, args.output_dir, args.device, languages=args.languages)
