#!/usr/bin/env python3
"""
TATR v1.1-All Table Structure Recognition Inference Script
Environment: tables-detr (Python 3.10)
"""
import sys
import os
import json
import argparse
from PIL import Image

repo_src = "/home/tanish/scratch/tanishq/table-understanding/repos/table-transformer/src"
sys.path.insert(0, repo_src)
os.chdir(repo_src)
from inference import TableExtractionPipeline

sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts")
from draw_bboxes import draw_bboxes_on_image

def run_tatr(image_path, device="cuda:0", output_dir=None):
    config_path = "/home/tanish/scratch/tanishq/table-understanding/repos/table-transformer/src/structure_config.json.pre-potatr"
    model_path = "/home/tanish/scratch/tanishq/table-understanding/repos/table-transformer/weights/TATR-v1.1-All-msft.pth"
    
    pipe = TableExtractionPipeline(
        str_config_path=config_path,
        str_model_path=model_path,
        str_device=device
    )
    img = Image.open(image_path).convert("RGB")
    
    import time
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
    
    # Cells in TATR are dicts with 'bbox' key
    cell_bboxes = []
    for c in cells_raw:
        if isinstance(c, dict) and "bbox" in c:
            cell_bboxes.append(c["bbox"])
        elif isinstance(c, (list, tuple)):
            cell_bboxes.append(c)

    res = {
        "model": "TATR v1.1-All",
        "html": html,
        "cell_bboxes": cell_bboxes,
        "num_cells": len(cell_bboxes),
        "raw_cells": cells_raw,
        "elapse_sec": elapse
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "tatr_raw.json"), "w") as f:
            json.dump(res, f, indent=2)
        with open(os.path.join(output_dir, "tatr.html"), "w") as f:
            f.write(html)
        if len(cell_bboxes) > 0:
            bbox_img_path = os.path.join(output_dir, "tatr_bboxes.png")
            draw_bboxes_on_image(image_path, cell_bboxes, bbox_img_path)
        print(f"Saved TATR outputs to {output_dir}")

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TATR v1.1-All inference")
    parser.add_argument("--image", type=str, required=True, help="Path to table image")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    args = parser.parse_args()

    result = run_tatr(args.image, device=args.device, output_dir=args.output_dir)
    print(f"TATR Detected cells: {result['num_cells']}")
    print(f"TATR HTML snippet: {result['html'][:250]}...")
