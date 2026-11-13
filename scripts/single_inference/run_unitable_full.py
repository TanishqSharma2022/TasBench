#!/usr/bin/env python3
"""
UniTable Complete Table Structure + Bounding Box Inference Script
Loads:
- unitable_large_structure.pt (Structure HTML Tokens with correct normalization)
- unitable_large_bbox.pt (Cell Bounding Box Coordinates)
Environment: unitable (Python 3.9)
"""
import sys
import os
import json
import argparse
import time
import re
from functools import partial
from typing import Sequence, Tuple, Optional

import torch
from torch import nn, Tensor
from PIL import Image
import torchvision.transforms as T
import tokenizers as tk

REPO_DIR = "/home/tanish/scratch/tanishq/table-understanding/repos/unitable"
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts")

from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder
from src.utils import subsequent_mask, pred_token_within_range, greedy_sampling, bbox_str_to_token_list
from src.trainer.utils import VALID_HTML_TOKEN, VALID_BBOX_TOKEN
from draw_bboxes import draw_bboxes_on_image

# Model constants
D_MODEL = 768
PATCH_SIZE = 16
NHEAD = 12
DROPOUT = 0.2

def build_model(vocab: tk.Tokenizer, max_seq_len: int, weight_path: str, device: torch.device):
    backbone = ImgLinearBackbone(d_model=D_MODEL, patch_size=PATCH_SIZE)
    encoder = Encoder(d_model=D_MODEL, nhead=NHEAD, dropout=DROPOUT, activation="gelu", norm_first=True, nlayer=12, ff_ratio=4)
    decoder = Decoder(d_model=D_MODEL, nhead=NHEAD, dropout=DROPOUT, activation="gelu", norm_first=True, nlayer=4, ff_ratio=4)
    model = EncoderDecoder(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab.get_vocab_size(),
        d_model=D_MODEL,
        padding_idx=vocab.token_to_id("<pad>"),
        max_seq_len=max_seq_len,
        dropout=DROPOUT,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def autoregressive_decode(model, memory, prefix, max_len, eos_id, whitelist, device):
    context = torch.tensor(prefix, dtype=torch.int32, device=device).unsqueeze(0)
    for _ in range(max_len):
        if eos_id in context[0]:
            break
        causal_mask = subsequent_mask(context.shape[1]).to(device)
        logits = model.decode(memory, context, tgt_mask=causal_mask, tgt_padding_mask=None)
        logits = model.generator(logits)[:, -1, :]
        logits = pred_token_within_range(logits.detach(), white_list=whitelist)
        _, next_tokens = greedy_sampling(logits)
        context = torch.cat([context, next_tokens], dim=1)
    return context[0]

def rescale_bbox(bbox: Sequence[Sequence[float]], src: Tuple[int, int], tgt: Tuple[int, int]):
    ratio = [tgt[0] / src[0], tgt[1] / src[1]] * 2
    return [[int(round(i * j)) for i, j in zip(entry, ratio)] for entry in bbox]

def run_unitable(image_path: str, device: str = "cuda:0", output_dir: Optional[str] = None):
    dev = torch.device(device)
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Official UniTable normalization
    transform = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean=[0.86597056, 0.88463002, 0.87491087], std=[0.20686628, 0.18201602, 0.18485524])
    ])
    src_tensor = transform(img).unsqueeze(0).to(dev)

    # 1. Structure Prediction
    struct_vocab_path = os.path.join(REPO_DIR, "vocab/vocab_html.json")
    struct_weight_path = os.path.join(REPO_DIR, "experiments/unitable_weights/unitable_large_structure.pt")
    struct_vocab = tk.Tokenizer.from_file(struct_vocab_path)
    struct_model = build_model(struct_vocab, 784, struct_weight_path, dev)
    
    struct_eos_id = struct_vocab.token_to_id("<eos>")
    struct_whitelist = [struct_vocab.token_to_id(tok) for tok in VALID_HTML_TOKEN] + [struct_eos_id]

    start_time = time.perf_counter()
    with torch.no_grad():
        struct_memory = struct_model.encode(src_tensor)
        struct_ids = autoregressive_decode(
            struct_model, struct_memory,
            [struct_vocab.token_to_id("[html]")],
            512, struct_eos_id, struct_whitelist, dev
        )
    struct_time = time.perf_counter() - start_time

    struct_tokens = [struct_vocab.id_to_token(tid) for tid in struct_ids.cpu().tolist()]
    html_body = "".join([t for t in struct_tokens if not t.startswith("[") and t not in ["<eos>", "<pad>", "<sos>"]])
    if not html_body.startswith("<table>"):
        html_body = f"<table>{html_body}</table>"

    # Free structure model memory
    del struct_model, struct_memory
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Bounding Box Prediction
    bbox_vocab_path = os.path.join(REPO_DIR, "vocab/vocab_bbox.json")
    bbox_weight_path = os.path.join(REPO_DIR, "experiments/unitable_weights/unitable_large_bbox.pt")
    bbox_vocab = tk.Tokenizer.from_file(bbox_vocab_path)
    bbox_model = build_model(bbox_vocab, 1024, bbox_weight_path, dev)

    bbox_eos_id = bbox_vocab.token_to_id("<eos>")
    bbox_whitelist = [bbox_vocab.token_to_id(i) for i in VALID_BBOX_TOKEN[:449]] + [bbox_eos_id]

    start_bbox = time.perf_counter()
    with torch.no_grad():
        bbox_memory = bbox_model.encode(src_tensor)
        bbox_ids = autoregressive_decode(
            bbox_model, bbox_memory,
            [bbox_vocab.token_to_id("[bbox]")],
            1024, bbox_eos_id, bbox_whitelist, dev
        )
    bbox_time = time.perf_counter() - start_bbox

    bbox_str = bbox_vocab.decode(bbox_ids.cpu().numpy(), skip_special_tokens=False)
    parsed_boxes_448 = bbox_str_to_token_list(bbox_str)
    scaled_boxes = rescale_bbox(parsed_boxes_448, src=(448, 448), tgt=(orig_w, orig_h))

    total_time = struct_time + bbox_time

    res = {
        "model": "UniTable",
        "tokens": struct_tokens,
        "html": html_body,
        "num_tokens": len(struct_tokens),
        "raw_bboxes_448": parsed_boxes_448,
        "cell_bboxes": scaled_boxes,
        "num_cells": len(scaled_boxes),
        "orig_image_size": [orig_w, orig_h],
        "elapse_sec": total_time,
        "structure_elapse_sec": struct_time,
        "bbox_elapse_sec": bbox_time
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "unitable_raw.json"), "w") as f:
            json.dump(res, f, indent=2)
        with open(os.path.join(output_dir, "unitable.html"), "w") as f:
            f.write(html_body)
        if len(scaled_boxes) > 0:
            bbox_img_path = os.path.join(output_dir, "unitable_bboxes.png")
            draw_bboxes_on_image(image_path, scaled_boxes, bbox_img_path)
        print(f"Saved UniTable outputs to {output_dir}")

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UniTable full inference")
    parser.add_argument("--image", type=str, required=True, help="Path to table image")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    args = parser.parse_args()

    result = run_unitable(args.image, device=args.device, output_dir=args.output_dir)
    print(f"UniTable Predicted {result['num_cells']} bboxes and {result['num_tokens']} tokens.")
    print(f"UniTable HTML: {result['html'][:250]}...")
