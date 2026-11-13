#!/usr/bin/env python3
"""
SmolDocling-256M Table Structure Recognition Inference Script
Environment: smoldocling (Python 3.10)
"""
import sys
import os
import json
import argparse
import time
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from docling_core.types.doc.document import DocTagsDocument
from docling_core.types.doc import DoclingDocument

sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts")
from draw_bboxes import draw_bboxes_on_image
from otsl_utils import otsl_to_html

def parse_doctag_bboxes(doctags, img_w, img_h):
    # Parse <loc_X><loc_Y><loc_X><loc_Y> tags normalized to 0-500
    boxes = []
    pattern = re.compile(r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")
    for match in pattern.finditer(doctags):
        y1, x1, y2, x2 = map(int, match.groups())
        # Rescale from 500x500 space
        x1_norm, y1_norm = x1 / 500.0, y1 / 500.0
        x2_norm, y2_norm = x2 / 500.0, y2 / 500.0
        boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
    return boxes

def run_smoldocling(image_path, model_id="ds4sd/SmolDocling-256M-preview", device="cuda:0", output_dir=None):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this table to docling."}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    s = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
    elapse = time.perf_counter() - s

    prompt_len = inputs.input_ids.shape[1]
    trimmed_ids = generated_ids[:, prompt_len:]
    doctags = processor.batch_decode(trimmed_ids, skip_special_tokens=False)[0].lstrip()

    # Generate HTML from doctags/OTSL
    html_text = otsl_to_html(doctags)
    bboxes = parse_doctag_bboxes(doctags, w, h)

    res = {
        "model": "SmolDocling-256M",
        "doctags": doctags,
        "html": html_text,
        "cell_bboxes": bboxes,
        "num_cells": len(bboxes),
        "elapse_sec": elapse
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "smoldocling_raw.json"), "w") as f:
            json.dump(res, f, indent=2)
        with open(os.path.join(output_dir, "smoldocling.html"), "w") as f:
            f.write(html_text)
        if len(bboxes) > 0:
            bbox_img_path = os.path.join(output_dir, "smoldocling_bboxes.png")
            draw_bboxes_on_image(image_path, bboxes, bbox_img_path)
        print(f"Saved SmolDocling outputs to {output_dir}")

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SmolDocling-256M inference")
    parser.add_argument("--image", type=str, required=True, help="Path to table image")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    args = parser.parse_args()

    result = run_smoldocling(args.image, device=args.device, output_dir=args.output_dir)
    print(f"SmolDocling Detected boxes: {result['num_cells']}")
    print(f"SmolDocling Doctags snippet: {result['doctags'][:250]}...")
