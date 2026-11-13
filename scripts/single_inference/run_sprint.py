#!/usr/bin/env python3
"""
SPRINT Table Structure Recognition Inference Script
Environment: sprint (Python 3.10)
"""
import sys
import os
import json
import argparse
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts")
from otsl_utils import otsl_to_html

def run_sprint(image_path, device="cuda:0", output_dir=None):
    model_weight_path = "/home/tanish/scratch/tanishq/table-understanding/repos/SPRINT/weights/SPRINT_merged.pt"
    dev = torch.device(device)
    
    # Load entire serialized model
    model = torch.load(model_weight_path, map_location=dev)
    model.to(dev)
    model.eval()

    test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    img = Image.open(image_path)
    img_tensor = test_transforms(img).unsqueeze(0).to(dev)

    import time
    s = time.perf_counter()
    with torch.no_grad():
        output = model(img_tensor, None, return_preds=True)
    elapse = time.perf_counter() - s

    otsl_pred = output["preds"][0][0]
    html_pred = otsl_to_html(otsl_pred)
    
    res = {
        "model": "SPRINT",
        "otsl": otsl_pred,
        "html": html_pred,
        "num_tokens": len(otsl_pred),
        "cell_bboxes": [],
        "elapse_sec": elapse
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "sprint_raw.json"), "w") as f:
            json.dump(res, f, indent=2)
        with open(os.path.join(output_dir, "sprint.html"), "w") as f:
            f.write(html_pred)
        print(f"Saved SPRINT outputs to {output_dir}")

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPRINT inference")
    parser.add_argument("--image", type=str, required=True, help="Path to table image")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    args = parser.parse_args()

    result = run_sprint(args.image, device=args.device, output_dir=args.output_dir)
    print(f"SPRINT OTSL Length: {result['num_tokens']}")
    print(f"SPRINT HTML snippet: {result['html'][:250]}...")
