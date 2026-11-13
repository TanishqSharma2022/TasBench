#!/usr/bin/env python3
"""
PIX-TAB Table Structure Recognition Inference Script
Environment: mtltabnet (Python 3.10)
Uses:
- epoch_25.pth (full-pubtabnet-500k-resume-epoch18)
- parse_papp_otsl_bboxes (derives spatial bboxes directly from PAPP-OTSL tokens)
- OTSLTableParser (generates valid multi-row HTML table with rowspans/colspans)
"""
import sys
import os
import json
import argparse
import time
import re
import torch
from PIL import Image

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

sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts")
from draw_bboxes import draw_bboxes_on_image

def run_pixtab(image_path, device="cuda:0", output_dir=None):
    config_file = f"{REPO_DIR}/full-pubtabnet-500k-resume-epoch18/table_master_ResnetExtract_Ranger_0705_cell150_batch4.py"
    checkpoint_file = "/home/tanish/scratch/tanishq/table-understanding/repos/PIX-TAB/PIX-TAB-organized/checkpoints/full-pubtabnet-500k-resume-epoch18/epoch_25.pth"

    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)
    if model.cfg.data.test.get("type") == "ConcatDataset":
        model.cfg.data.test.pipeline = model.cfg.data.test["datasets"][0].pipeline

    s = time.perf_counter()
    result = model_inference(model, image_path)
    elapse = time.perf_counter() - s

    orig_img = Image.open(image_path)
    orig_w, orig_h = orig_img.size

    tokens_str = result.get("text", "")

    # 1. Parse HTML via OTSLTableParser
    parser = OTSLTableParser(tokens_str)
    parsed = parser.parse()
    clean_html = re.sub(r"Cell_\d+_\d+", "", parsed["html"])
    clean_html = re.sub(r' data-bbox=\"\[.*?\]\"', "", clean_html)
    clean_html = "".join(clean_html.split("\n"))

    # 2. Derive Bounding Boxes via official parse_papp_otsl_bboxes
    papp_bboxes_dict = parse_papp_otsl_bboxes(tokens_str, (orig_w, orig_h), model_input_size=(480, 480))
    cell_bboxes = [item["bbox"] for item in papp_bboxes_dict]

    res = {
        "model": "PIX-TAB",
        "checkpoint": "epoch_25.pth",
        "html": clean_html,
        "tokens": tokens_str,
        "cell_bboxes_meta": papp_bboxes_dict,
        "cell_bboxes": cell_bboxes,
        "num_cells": len(cell_bboxes),
        "orig_image_size": [orig_w, orig_h],
        "elapse_sec": elapse
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "pixtab_raw.json"), "w") as f:
            json.dump(res, f, indent=2)
        with open(os.path.join(output_dir, "pixtab.html"), "w") as f:
            f.write(clean_html)
        if len(cell_bboxes) > 0:
            bbox_img_path = os.path.join(output_dir, "pixtab_bboxes.png")
            draw_bboxes_on_image(image_path, cell_bboxes, bbox_img_path)
        print(f"Saved PIX-TAB outputs to {output_dir}")

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PIX-TAB inference")
    parser.add_argument("--image", type=str, required=True, help="Path to table image")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    args = parser.parse_args()

    result = run_pixtab(args.image, device=args.device, output_dir=args.output_dir)
    print(f"PIX-TAB Detected cells: {result['num_cells']}")
    print(f"PIX-TAB HTML snippet: {result['html'][:250]}...")
