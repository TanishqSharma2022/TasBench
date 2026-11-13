#!/usr/bin/env python3
"""
Batch Inference for SPRINT with docTR Normalization and TATR Grid Alignment.
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

ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

def normalize_sprint_otsl(s):
    res = []
    for ch in s:
        if ch in ["E", "F", "C"]:
            res.append("C")
        elif ch in ["L", "U", "X", "N"]:
            res.append(ch)
        else:
            res.append("C")
    return "".join(res)

def align_otsl_from_rows_cols(otsl_string, rows, cols):
    N = len(otsl_string)
    C = cols
    R = rows
    actual_N = (C + 1) * R
    if N != actual_N:
        if N > actual_N:
            otsl_string = otsl_string[:actual_N]
            otsl_string = otsl_string[:-1] + "N"
        else:
            diff = actual_N - N
            otsl_string = otsl_string + "C" * (diff - 1) + "N"

    otsl_string_list = []
    for i in range(len(list(otsl_string))):
        char = otsl_string[i]
        if i > 0 and (i + 1) % (C + 1) == 0:
            char = "N"
        else:
            if otsl_string[i] == "N":
                char = "C"
        otsl_string_list.append(char)
    return "".join(otsl_string_list)

def count_contiguous_occurrences(s, target_char):
    count = 0
    for char in s:
        if char == target_char:
            count += 1
        else:
            break
    return count

def get_cell_spans(otsl_matrix, i, j):
    entry = otsl_matrix[i][j]
    if entry != "C":
        return 0, 0
    else:
        row_seq = "".join(otsl_matrix[i])[j + 1:]
        col_seq = "".join(row[j] for row in otsl_matrix)[i + 1:]
        rs = count_contiguous_occurrences(row_seq, "L")
        cs = count_contiguous_occurrences(col_seq, "U")
        return rs, cs

def get_conv_html_from_otsl(otsl_matrix, R, C):                
    html_string = "<table><tbody>"
    for i in range(R):
        html_string += "<tr>"
        for j in range(C + 1):
            e = otsl_matrix[i][j]
            if e == "C":
                rs, cs = get_cell_spans(otsl_matrix, i, j)
                if rs and cs:
                    html_string += f'<td rowspan="{cs + 1}" colspan="{rs + 1}"></td>'
                elif rs and not cs:
                    html_string += f'<td colspan="{rs + 1}"></td>'
                elif not rs and cs:
                    html_string += f'<td rowspan="{cs + 1}"></td>'
                else:
                    html_string += "<td></td>"
            elif e == "N":
                html_string += "</tr>"
            else:
                continue
    html_string += "</tbody></table>"
    return html_string

def convert_to_html(otsl_string, R, C):
    actual_N = (C + 1) * R
    if len(otsl_string) != actual_N:
        if len(otsl_string) > actual_N:
            otsl_string = otsl_string[:actual_N]
            otsl_string = otsl_string[:-1] + "N"
        else:
            diff = actual_N - len(otsl_string)
            otsl_string = otsl_string + "C" * (diff - 1) + "N"
    
    otsl_matrix = [[otsl_string[i * (C + 1) + j] for j in range(C + 1)] for i in range(R)]

    for i in range(len(otsl_matrix[0])):
        if otsl_matrix[0][i] == "U":
            otsl_matrix[0][i] = "C"
    for i in range(R):
        if otsl_matrix[i][0] == "L":
            otsl_matrix[i][0] = "C"

    return get_conv_html_from_otsl(otsl_matrix, R, C)

def fallback_otsl_to_html(otsl_text):
    norm = normalize_sprint_otsl(otsl_text)
    if not norm:
        return "<table><tbody></tbody></table>"
    rows_raw = norm.strip().split("N")
    rows = [list(r) for r in rows_raw if r]
    if not rows:
        return "<table><tbody></tbody></table>"
    R = len(rows)
    C = max(len(r) for r in rows) if rows else 0
    if C == 0:
        return "<table><tbody></tbody></table>"
    grid = []
    for r in rows:
        if len(r) < C:
            r = r + ["C"] * (C - len(r))
        grid.append(r[:C])
    
    html = ["<table><tbody>"]
    for i in range(R):
        html.append("<tr>")
        for j in range(C):
            cell = grid[i][j]
            if cell == "C":
                colspan = 1
                while j + colspan < C and grid[i][j + colspan] == "L":
                    colspan += 1
                rowspan = 1
                while i + rowspan < R and grid[i + rowspan][j] == "U":
                    rowspan += 1
                cs_str = f' colspan="{colspan}"' if colspan > 1 else ""
                rs_str = f' rowspan="{rowspan}"' if rowspan > 1 else ""
                html.append(f"<td{rs_str}{cs_str}></td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)

def run_batch(data_root: Path, output_dir: Path, tatr_grids_file: Path, device: str = "cuda:3", languages=None, batch_size: int = 8):
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SPRINT] Loading model on {device}...")
    model_weight_path = "/home/tanish/scratch/tanishq/table-understanding/repos/SPRINT/weights/SPRINT_merged.pt"
    dev = torch.device(device)
    model = torch.load(model_weight_path, map_location=dev)
    model.to(dev)
    model.eval()

    # docTR Normalization as per paper and model.cfg
    test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))
    ])

    print(f"[SPRINT] Loading TATR grid dimensions from {tatr_grids_file}...")
    with open(tatr_grids_file, "r") as fp:
        all_tatr_grids = json.load(fp)

    target_languages = languages or ALL_LANGUAGES
    for lang in target_languages:
        ann_file = data_root / "annotations" / f"{lang}.jsonl"
        with open(ann_file, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]

        out_file = preds_dir / f"{lang}.jsonl"
        
        # Check if already completed
        if out_file.exists():
            with open(out_file, "r") as f:
                existing = [line for line in f if line.strip()]
            if len(existing) >= len(items):
                print(f"[SPRINT] {lang} already completed ({len(existing)} records). Skipping.")
                continue

        lang_tatr_grids = all_tatr_grids.get(lang, {})
        print(f"[SPRINT] Starting inference for {lang} ({len(items)} samples) on {device}...")

        records = []
        with open(out_file, "w", encoding="utf-8") as f_out:
            for b_idx in tqdm(range(0, len(items), batch_size), desc=f"SPRINT [{lang}]"):
                batch_items = items[b_idx: b_idx + batch_size]
                batch_tensors = []
                for it in batch_items:
                    img_path = data_root / it["image"]
                    try:
                        img = Image.open(str(img_path))
                        tensor = test_transforms(img)
                    except Exception as e:
                        tensor = torch.zeros(3, 128, 128)
                    batch_tensors.append(tensor)

                batch_input = torch.stack(batch_tensors).to(dev)

                s = time.perf_counter()
                with torch.no_grad():
                    output = model(batch_input, None, return_preds=True)
                elapse = (time.perf_counter() - s) / len(batch_items)

                for idx, it in enumerate(batch_items):
                    img_id = it["image_id"]
                    bucket = it.get("complexity_bucket", "unknown")
                    try:
                        raw_otsl = output["preds"][idx][0]
                        norm_otsl = normalize_sprint_otsl(raw_otsl)
                        
                        # TATR grid alignment
                        grid = lang_tatr_grids.get(img_id, [0, 0])
                        R, C = grid[0], grid[1]

                        if R > 0 and C > 0:
                            aligned_otsl = align_otsl_from_rows_cols(norm_otsl, R, C)
                            html_pred = convert_to_html(aligned_otsl, R, C)
                        else:
                            html_pred = fallback_otsl_to_html(norm_otsl)

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
                            "elapse_sec": elapse,
                            "status": "error",
                            "error": str(e)
                        }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_out.flush()
                    records.append(rec)

        print(f"[SPRINT] Finished {lang} ({len(records)} records).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tatr-grids", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--language", action="append", dest="languages", default=None)
    args = parser.parse_args()

    run_batch(
        args.data_root,
        args.output_dir,
        args.tatr_grids,
        args.device,
        languages=args.languages,
        batch_size=args.batch_size
    )
