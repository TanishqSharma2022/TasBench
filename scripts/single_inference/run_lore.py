#!/usr/bin/env python3
"""
LORE-TSR Table Structure Recognition Inference Script
Environment: lore-tsr (Python 3.10)
"""
import sys
import os
import json
import argparse
from lineless_table_rec import LinelessTableRecognition
from lineless_table_rec.main import LinelessTableInput

sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts")
from draw_bboxes import draw_bboxes_on_image

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

def run_lore(image_path, use_cuda=True, output_dir=None):
    cfg = LinelessTableInput(model_type="lore", use_cuda=use_cuda)
    engine = LinelessTableRecognition(cfg)
    
    # Pass empty ocr_result to evaluate pure structure
    out = engine(image_path, ocr_result=[])
    
    bboxes = out.cell_bboxes.tolist() if hasattr(out.cell_bboxes, "tolist") else (out.cell_bboxes or [])
    logic_pts = out.logic_points.tolist() if hasattr(out.logic_points, "tolist") else (out.logic_points or [])
    html = logic_points_to_html(logic_pts) if logic_pts else out.pred_html

    res = {
        "model": "LORE-TSR",
        "html": html,
        "cell_bboxes": bboxes,
        "logic_points": logic_pts,
        "num_cells": len(bboxes),
        "elapse_sec": out.elapse
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "lore_raw.json"), "w") as f:
            json.dump(res, f, indent=2)
        with open(os.path.join(output_dir, "lore.html"), "w") as f:
            f.write(html)
        if len(bboxes) > 0:
            bbox_img_path = os.path.join(output_dir, "lore_bboxes.png")
            draw_bboxes_on_image(image_path, bboxes, bbox_img_path)
        print(f"Saved LORE outputs to {output_dir}")

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LORE-TSR inference")
    parser.add_argument("--image", type=str, required=True, help="Path to table image")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    args = parser.parse_args()

    result = run_lore(args.image, use_cuda=not args.cpu, output_dir=args.output_dir)
    print(f"LORE Detected cells: {result['num_cells']}")
    print(f"LORE HTML snippet: {result['html'][:250]}...")
