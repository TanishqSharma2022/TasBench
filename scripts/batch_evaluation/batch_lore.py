#!/usr/bin/env python3
"""
Batch Inference for LORE-TSR.
Environment: lore-tsr
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm

from lineless_table_rec import LinelessTableRecognition
from lineless_table_rec.main import LinelessTableInput

ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

def logic_points_to_html(logic_points):
    if logic_points is None or len(logic_points) == 0:
        return "<table></table>"
    max_row = max(p[1] for p in logic_points) + 1
    max_col = max(p[3] for p in logic_points) + 1
    grid = [[None] * max_col for _ in range(max_row)]
    for i, (r1, r2, c1, c2) in enumerate(logic_points):
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                grid[r][c] = (i, r1, r2, c1, c2)
    html = ["<table>"]
    for r in range(max_row):
        row_str = ["<tr>"]
        for c in range(max_col):
            if grid[r][c] is not None:
                i, r1, r2, c1, c2 = grid[r][c]
                if r == r1 and c == c1:
                    rs = f' rowspan="{r2 - r1 + 1}"' if r2 > r1 else ""
                    cs = f' colspan="{c2 - c1 + 1}"' if c2 > c1 else ""
                    row_str.append(f"<td{rs}{cs}></td>")
            else:
                row_str.append("<td></td>")
        row_str.append("</tr>")
        html.append("".join(row_str))
    html.append("</table>")
    return "".join(html)

def run_batch(data_root: Path, output_dir: Path, use_cuda: bool = True, languages=None):
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LORE-TSR] Initializing engine (use_cuda={use_cuda})...")
    cfg = LinelessTableInput(model_type="lore", use_cuda=use_cuda)
    engine = LinelessTableRecognition(cfg)

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

                print(f"[LORE-TSR] {lang} already completed ({len(existing)} records). Skipping.")

                continue

        records = []
        for item in tqdm(items, desc=f"LORE [{lang}]"):
            img_rel = item["image"]
            img_path = str(data_root / img_rel)
            img_id = item["image_id"]
            bucket = item.get("complexity_bucket", "unknown")

            try:
                out = engine(img_path, ocr_result=[])
                bboxes = out.cell_bboxes.tolist() if hasattr(out.cell_bboxes, "tolist") else (out.cell_bboxes or [])
                logic_pts = out.logic_points.tolist() if hasattr(out.logic_points, "tolist") else (out.logic_points or [])
                html = logic_points_to_html(logic_pts) if logic_pts else out.pred_html

                rec = {
                    "image_id": img_id,
                    "language": lang,
                    "complexity_bucket": bucket,
                    "prediction_html": html,
                    "cell_bboxes": bboxes,
                    "elapse_sec": out.elapse,
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
        print(f"[LORE-TSR] Finished {lang} ({len(records)} records).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--language", action="append", dest="languages", default=None)
    args = parser.parse_args()

    run_batch(args.data_root, args.output_dir, use_cuda="cuda" in args.device, languages=args.languages)
