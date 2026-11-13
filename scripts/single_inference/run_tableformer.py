#!/usr/bin/env python3
"""
TableFormer V2 Table Structure Recognition Inference Script
Environment: tableformer (Python 3.10)
"""
import sys
import os
import json
import argparse
import time
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
from docling_ibm_models.tableformer_v2 import TableFormerV2

sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts")
from draw_bboxes import draw_bboxes_on_image
from otsl_utils import otsl_to_html

def run_tableformer(image_path, model_id="docling-project/TableFormerV2", device="cuda:0", output_dir=None):
    dev = torch.device(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = TableFormerV2.from_pretrained(model_id).to(dev)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(dev)

    s = time.perf_counter()
    with torch.no_grad():
        out = model.generate(img_tensor, tokenizer=tokenizer, max_length=512)
    elapse = time.perf_counter() - s

    gen_tokens = tokenizer.decode(out["generated_ids"][0], skip_special_tokens=True)
    bboxes = out["predicted_bboxes"][0].cpu().numpy().tolist() if out["predicted_bboxes"] is not None else []
    html_text = otsl_to_html(gen_tokens)

    res = {
        "model": "TableFormerV2",
        "otsl_tokens": gen_tokens,
        "html": html_text,
        "cell_bboxes": bboxes,
        "num_cells": len(bboxes),
        "elapse_sec": elapse
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "tableformer_raw.json"), "w") as f:
            json.dump(res, f, indent=2)
        with open(os.path.join(output_dir, "tableformer.html"), "w") as f:
            f.write(html_text)
        if len(bboxes) > 0:
            bbox_img_path = os.path.join(output_dir, "tableformer_bboxes.png")
            draw_bboxes_on_image(image_path, bboxes, bbox_img_path)
        print(f"Saved TableFormer outputs to {output_dir}")

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TableFormerV2 inference")
    parser.add_argument("--image", type=str, required=True, help="Path to table image")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    args = parser.parse_args()

    result = run_tableformer(args.image, device=args.device, output_dir=args.output_dir)
    print(f"TableFormer Detected cells: {result['num_cells']}")
    print(f"TableFormer OTSL: {result['otsl_tokens'][:250]}...")
