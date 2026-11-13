#!/usr/bin/env python3
"""
TFLOP Table Structure Recognition Inference Script
Environment: tflop (Python 3.10)
"""
import sys
import os
import json
import argparse
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

repo_dir = "/home/tanish/scratch/tanishq/table-understanding/repos/TFLOP"
sys.path.insert(0, repo_dir)
from tflop.model.model.TFLOP import TFLOP
from tflop.model.model.TFLOP_Config import TFLOPConfig

sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts")
from otsl_utils import otsl_to_html

def run_tflop(image_path, device="cuda:0", output_dir=None):
    weights_dir = os.path.join(repo_dir, "weights/upstage-TFLOP")
    with open(os.path.join(weights_dir, "config.json")) as f:
        cfg_dict = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(weights_dir)
    config = TFLOPConfig(**cfg_dict)
    model = TFLOP(config=config, tokenizer=tokenizer, data_ids=["C-tag"])

    saved_state_dict = torch.load(os.path.join(weights_dir, "pytorch_model.bin"), map_location="cpu")
    encoder_state_dict = {k[len("encoder."):]: v for k, v in saved_state_dict.items() if k.startswith("encoder.")}
    decoder_state_dict = {k[len("decoder."):]: v for k, v in saved_state_dict.items() if k.startswith("decoder.")}
    model.encoder.load_state_dict(encoder_state_dict)
    model.decoder.load_state_dict(decoder_state_dict)

    dev = torch.device(device)
    model.to(dev).bfloat16()
    model.eval()

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((config.input_size[0], config.input_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(dev).bfloat16()

    prompt_ids = tokenizer("<s><s_answer>", add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
    coords_int_padded = torch.full((1, config.bbox_token_cnt, 4), 771, dtype=torch.long, device=dev)
    pointer_args = {
        "coord_input_idx": coords_int_padded,
        "coord_input_length": torch.tensor([1], dtype=torch.long, device=dev)
    }

    import time
    s = time.perf_counter()
    with torch.no_grad():
        out = model.inference(
            image_tensors=img_tensor,
            prompt_tensors=prompt_ids,
            return_json=False,
            pointer_args=pointer_args
        )
    elapse = time.perf_counter() - s

    gen_seq = out["output_sequences"][0]
    tokens_decoded = tokenizer.decode(gen_seq, skip_special_tokens=False)
    html_text = otsl_to_html(tokens_decoded)

    res = {
        "model": "TFLOP",
        "output_tokens": tokens_decoded,
        "html": html_text,
        "num_tokens": len(gen_seq),
        "cell_bboxes": [],
        "elapse_sec": elapse
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "tflop_raw.json"), "w") as f:
            json.dump(res, f, indent=2)
        with open(os.path.join(output_dir, "tflop.html"), "w") as f:
            f.write(html_text)
        print(f"Saved TFLOP outputs to {output_dir}")

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TFLOP inference")
    parser.add_argument("--image", type=str, required=True, help="Path to table image")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    args = parser.parse_args()

    result = run_tflop(args.image, device=args.device, output_dir=args.output_dir)
    print(f"TFLOP Generated tokens count: {result['num_tokens']}")
    print(f"TFLOP Decoded snippet: {result['output_tokens'][:250]}...")
