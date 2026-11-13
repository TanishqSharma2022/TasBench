#!/usr/bin/env python3
"""
UniTable Table Structure Recognition Inference Script
Environment: unitable (Python 3.10)
"""
import sys
import os
import json
import argparse
import torch
from functools import partial
from torch import nn, Tensor
from PIL import Image
import torchvision.transforms as T
import tokenizers as tk

sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/unitable")
from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder
from src.utils import subsequent_mask, pred_token_within_range, greedy_sampling
from src.trainer.utils import VALID_HTML_TOKEN

def run_unitable(image_path, device="cuda:0", output_dir=None):
    vocab_path = "/home/tanish/scratch/tanishq/table-understanding/repos/unitable/vocab/vocab_html.json"
    weight_path = "/home/tanish/scratch/tanishq/table-understanding/repos/unitable/experiments/unitable_weights/unitable_large_structure.pt"
    
    d_model = 768
    patch_size = 16
    nhead = 12
    dropout = 0.2
    max_seq_len = 784

    backbone = ImgLinearBackbone(d_model=d_model, patch_size=patch_size)
    encoder = Encoder(d_model=d_model, nhead=nhead, dropout=dropout, activation="gelu", norm_first=True, nlayer=12, ff_ratio=4)
    decoder = Decoder(d_model=d_model, nhead=nhead, dropout=dropout, activation="gelu", norm_first=True, nlayer=4, ff_ratio=4)

    vocab = tk.Tokenizer.from_file(vocab_path)
    model = EncoderDecoder(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab.get_vocab_size(),
        d_model=d_model,
        padding_idx=vocab.token_to_id("<pad>"),
        max_seq_len=max_seq_len,
        dropout=dropout,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )

    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    transform = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    src = transform(img).unsqueeze(0).to(dev)

    eos_id = vocab.token_to_id("<eos>")
    token_whitelist = [vocab.token_to_id(tok) for tok in VALID_HTML_TOKEN] + [eos_id]

    import time
    s = time.perf_counter()
    with torch.no_grad():
        memory = model.encode(src)
        prefix = [vocab.token_to_id("[html]")]
        context = torch.tensor(prefix, dtype=torch.int32, device=dev).unsqueeze(0)

        for _ in range(512):
            if eos_id in context[0]:
                break
            causal_mask = subsequent_mask(context.shape[1]).to(dev)
            logits = model.decode(memory, context, tgt_mask=causal_mask, tgt_padding_mask=None)
            logits = model.generator(logits)[:, -1, :]
            logits = pred_token_within_range(logits.detach(), white_list=token_whitelist)
            _, next_tokens = greedy_sampling(logits)
            context = torch.cat([context, next_tokens], dim=1)
    elapse = time.perf_counter() - s

    decoded_tokens = [vocab.id_to_token(tid) for tid in context[0].cpu().tolist()]
    html = "".join([t for t in decoded_tokens if not t.startswith("[") and t not in ["<eos>", "<pad>", "<sos>"]])
    if not html.startswith("<table>"):
        html = f"<table>{html}</table>"

    res = {
        "model": "UniTable",
        "tokens": decoded_tokens,
        "html": html,
        "num_tokens": len(decoded_tokens),
        "cell_bboxes": [],
        "elapse_sec": elapse
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "unitable_raw.json"), "w") as f:
            json.dump(res, f, indent=2)
        with open(os.path.join(output_dir, "unitable.html"), "w") as f:
            f.write(html)
        print(f"Saved outputs to {output_dir}")

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UniTable inference")
    parser.add_argument("--image", type=str, required=True, help="Path to table image")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    args = parser.parse_args()

    result = run_unitable(args.image, device=args.device, output_dir=args.output_dir)
    print(f"UniTable Tokens: {result['num_tokens']}")
    print(f"UniTable HTML: {result['html'][:250]}...")
