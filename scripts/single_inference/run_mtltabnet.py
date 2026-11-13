#!/usr/bin/env python3
"""
MTL-TabNet Table Structure Recognition Inference Script
Environment: mtltabnet (Python 3.10)
"""
import sys
import os
import json
import argparse
import time
import numpy as np
import torch
from PIL import Image

REPO_DIR = "/home/tanish/scratch/tanishq/table-understanding/repos/MTL-TabNet"
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

import mmocr.models
import mmocr.datasets
import mmocr.datasets.pipelines
from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference

sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts")
from draw_bboxes import draw_bboxes_on_image

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

def run_mtltabnet(image_path, device="cuda:0", output_dir=None):
    config_file = f"{REPO_DIR}/configs/textrecog/master/table_master_ResnetExtract_Ranger_0705.py"
    checkpoint_file = f"{REPO_DIR}/table_recognition/models/PubTabNet/epoch_19.pth"

    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)
    if model.cfg.data.test.get("type") == "ConcatDataset":
        model.cfg.data.test.pipeline = model.cfg.data.test["datasets"][0].pipeline

    s = time.perf_counter()
    result = model_inference(model, image_path)
    elapse = time.perf_counter() - s

    orig_img = Image.open(image_path)
    orig_w, orig_h = orig_img.size

    tokens_str = result.get("text", "")
    raw_bboxes = np.array(result.get("bbox", []))

    # MTL-TabNet convertor output_format already scales coordinates to original image space
    # The format is [cx, cy, w, h] (center x, center y, width, height)
    # Convert cxcywh to xyxy:
    cell_bboxes = []
    if len(raw_bboxes) > 0:
        new_bboxes = np.empty_like(raw_bboxes)
        new_bboxes[..., 0] = raw_bboxes[..., 0] - raw_bboxes[..., 2] / 2.0
        new_bboxes[..., 1] = raw_bboxes[..., 1] - raw_bboxes[..., 3] / 2.0
        new_bboxes[..., 2] = raw_bboxes[..., 0] + raw_bboxes[..., 2] / 2.0
        new_bboxes[..., 3] = raw_bboxes[..., 1] + raw_bboxes[..., 3] / 2.0

        for b in new_bboxes:
            # Filter non-cell zero boxes
            if float(b[2] - b[0]) > 1.0 and float(b[3] - b[1]) > 1.0 and b.sum() != 0:
                x1 = max(0.0, min(float(orig_w), float(b[0])))
                y1 = max(0.0, min(float(orig_h), float(b[1])))
                x2 = max(0.0, min(float(orig_w), float(b[2])))
                y2 = max(0.0, min(float(orig_h), float(b[3])))
                cell_bboxes.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])

    html_text = clean_mtltabnet_tokens(tokens_str)

    res = {
        "model": "MTL-TabNet",
        "html": html_text,
        "tokens": tokens_str,
        "raw_bboxes": raw_bboxes.tolist(),
        "cell_bboxes": cell_bboxes,
        "num_cells": len(cell_bboxes),
        "orig_image_size": [orig_w, orig_h],
        "elapse_sec": elapse
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "mtltabnet_raw.json"), "w") as f:
            json.dump(res, f, indent=2)
        with open(os.path.join(output_dir, "mtltabnet.html"), "w") as f:
            f.write(html_text)
        if len(cell_bboxes) > 0:
            bbox_img_path = os.path.join(output_dir, "mtltabnet_bboxes.png")
            draw_bboxes_on_image(image_path, cell_bboxes, bbox_img_path)
        print(f"Saved MTL-TabNet outputs to {output_dir}")

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MTL-TabNet inference")
    parser.add_argument("--image", type=str, required=True, help="Path to table image")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    args = parser.parse_args()

    result = run_mtltabnet(args.image, device=args.device, output_dir=args.output_dir)
    print(f"MTL-TabNet Detected cells: {result['num_cells']}")
    print(f"MTL-TabNet HTML snippet: {result['html'][:250]}...")
