#!/usr/bin/env python3
"""
Multi-GPU UniTable Cell Bounding Box Extractor.
Shards inference across GPUs 0, 1, 2, 3, 6, 7 to extract cell_bboxes for all 11 languages.
Updates unitable/predictions/<lang>.jsonl with predicted cell_bboxes.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T
import tokenizers as tk
from functools import partial
from torch import nn
from tqdm import tqdm

DATA_ROOT = Path("/scratch/tanish/tanishq/table-understanding/data/Table Datasets/pubtabnet/multilingual_500_master_run")
EVAL_ROOT = Path("/scratch/tanish/tanishq/table-understanding/repos/synthetic-table-generation/eval_multilingual_500_master_run")
UNITABLE_ROOT = Path("/scratch/tanish/tanishq/table-understanding/repos/unitable")

sys.path.insert(0, str(UNITABLE_ROOT))
from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder
from src.utils import subsequent_mask, pred_token_within_range, greedy_sampling, bbox_str_to_token_list
from src.trainer.utils import VALID_BBOX_TOKEN

def rescale_bbox(bbox, src, tgt):
    ratio = [tgt[0] / src[0], tgt[1] / src[1]] * 2
    return [[round(float(i * j), 2) for i, j in zip(entry, ratio)] for entry in bbox]

def run_worker(gpu_id: int, tasks: list, out_dir: Path):
    dev = torch.device(f"cuda:{gpu_id}")
    vocab_path = UNITABLE_ROOT / "vocab" / "vocab_bbox.json"
    weight_path = UNITABLE_ROOT / "experiments" / "unitable_weights" / "unitable_large_bbox.pt"

    vocab = tk.Tokenizer.from_file(str(vocab_path))
    backbone = ImgLinearBackbone(d_model=768, patch_size=16)
    encoder = Encoder(d_model=768, nhead=12, dropout=0.2, activation="gelu", norm_first=True, nlayer=12, ff_ratio=4)
    decoder = Decoder(d_model=768, nhead=12, dropout=0.2, activation="gelu", norm_first=True, nlayer=4, ff_ratio=4)

    model = EncoderDecoder(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab.get_vocab_size(),
        d_model=768,
        padding_idx=vocab.token_to_id("<pad>"),
        max_seq_len=1024,
        dropout=0.2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.to(dev)
    model.eval()

    transform = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean=[0.865, 0.865, 0.865], std=[0.279, 0.279, 0.279])
    ])

    token_whitelist = [vocab.token_to_id(i) for i in VALID_BBOX_TOKEN[:449]]
    prefix_id = vocab.token_to_id("[bbox]")
    eos_id = vocab.token_to_id("<eos>")

    for lang, img_id, img_rel in tqdm(tasks, desc=f"GPU {gpu_id}"):
        img_path = DATA_ROOT / img_rel
        try:
            im = Image.open(img_path).convert("RGB")
            im_size = im.size
            img_tensor = transform(im).unsqueeze(0).to(dev)

            with torch.no_grad():
                memory = model.encode(img_tensor)
                context = torch.tensor([[prefix_id]], dtype=torch.long, device=dev)
                for _ in range(1024):
                    if eos_id in context[0]:
                        break
                    causal_mask = subsequent_mask(context.shape[1]).to(dev)
                    logits = model.decode(memory, context, tgt_mask=causal_mask, tgt_padding_mask=None)
                    logits = model.generator(logits)[:, -1, :]
                    logits = pred_token_within_range(logits.detach(), white_list=token_whitelist)
                    _, next_tokens = greedy_sampling(logits)
                    context = torch.cat([context, next_tokens], dim=1)

            raw_tokens = vocab.decode(context[0].cpu().numpy(), skip_special_tokens=False)
            raw_boxes = bbox_str_to_token_list(raw_tokens)
            scaled_boxes = rescale_bbox(raw_boxes, src=(448, 448), tgt=im_size)
        except Exception as e:
            scaled_boxes = []

        # Save result to shard file
        shard_file = out_dir / f"boxes_{lang}_{gpu_id}.jsonl"
        with open(shard_file, "a", encoding="utf-8") as fp:
            fp.write(json.dumps({"image_id": img_id, "language": lang, "cell_bboxes": scaled_boxes}) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.task_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    run_worker(args.gpu, tasks, args.out_dir)
