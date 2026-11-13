#!/usr/bin/env python3
"""
SLANet-plus Table Structure Recognition Inference Script
Environment: slanet (Python 3.10)
"""
import sys
import os
import json
import argparse
import cv2
from rapid_table import RapidTable, RapidTableInput, ModelType

sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts")
from draw_bboxes import draw_bboxes_on_image

def run_slanet(image_path, use_cuda=True, output_dir=None):
    cfg = RapidTableInput(model_type=ModelType.SLANETPLUS, use_ocr=False)
    engine = RapidTable(cfg)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    import time
    s = time.perf_counter()
    pred_html_tokens, cell_bboxes = engine.table_structure([img])
    elapse = time.perf_counter() - s
    
    tokens = pred_html_tokens[0][0] if pred_html_tokens and len(pred_html_tokens[0]) > 0 else []
    bboxes = cell_bboxes[0].tolist() if len(cell_bboxes) > 0 and len(cell_bboxes[0]) > 0 else []
    
    html = "".join(tokens)
    if not html.startswith("<table>"):
        html = f"<table>{html}</table>"

    res = {
        "model": "SLANet-plus",
        "html": html,
        "tokens": tokens,
        "cell_bboxes": bboxes,
        "num_cells": len(bboxes),
        "elapse_sec": elapse
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "slanet_raw.json"), "w") as f:
            json.dump(res, f, indent=2)
        with open(os.path.join(output_dir, "slanet.html"), "w") as f:
            f.write(html)
        if len(bboxes) > 0:
            bbox_img_path = os.path.join(output_dir, "slanet_bboxes.png")
            draw_bboxes_on_image(image_path, bboxes, bbox_img_path)
        print(f"Saved SLANet-plus outputs to {output_dir}")

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLANet-plus inference")
    parser.add_argument("--image", type=str, required=True, help="Path to table image")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    args = parser.parse_args()

    result = run_slanet(args.image, use_cuda=not args.cpu, output_dir=args.output_dir)
    print(f"SLANet Detected cells: {result['num_cells']}")
    print(f"SLANet HTML snippet: {result['html'][:250]}...")
