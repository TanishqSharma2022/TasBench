#!/usr/bin/env python3
"""
Batch Inference for UniTable.
Environment: unitable
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from functools import partial
import torch
from torch import nn
from PIL import Image
import torchvision.transforms as T
import tokenizers as tk
from tqdm import tqdm

sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/unitable")
from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder
from src.utils import subsequent_mask, pred_token_within_range, greedy_sampling
from src.trainer.utils import VALID_HTML_TOKEN

ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

def run_batch(data_root: Path, output_dir: Path, device: str = "cuda:0", languages=None):
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"[UniTable] Loading model on {device}...")
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
        T.Normalize(mean=[0.865, 0.865, 0.865], std=[0.279, 0.279, 0.279])
    ])

    token_whitelist = [vocab.token_to_id(t) for t in VALID_HTML_TOKEN if vocab.token_to_id(t) is not None]
    sos_id = vocab.token_to_id("<sos>")
    eos_id = vocab.token_to_id("<eos>")

    target_languages = languages or ALL_LANGUAGES
    for lang in target_languages:
        ann_file = data_root / "annotations" / f"{lang}.jsonl"

        with open(ann_file, "r", encoding="utf-8") as f:

            items = [json.loads(line) for line in f if line.strip()]


        out_file = preds_dir / f"{lang}.jsonl"
        processed_ids = set()
        if out_file.exists():
            with open(out_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            processed_ids.add(json.loads(line)["image_id"])
                        except Exception:
                            pass

        if len(processed_ids) >= len(items):
            print(f"[UniTable] {lang} already completed ({len(processed_ids)} records). Skipping.")
            continue

        records = []
        for item in tqdm(items, desc=f"UniTable [{lang}]"):
            if item["image_id"] in processed_ids:
                continue
            img_rel = item["image"]
            img_path = str(data_root / img_rel)
            img_id = item["image_id"]
            bucket = item.get("complexity_bucket", "unknown")

            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(dev)

                s = time.perf_counter()
                with torch.no_grad():
                    memory = model.encode(img_tensor)
                    context = torch.tensor([[sos_id]], dtype=torch.long, device=dev)
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

                rec = {
                    "image_id": img_id,
                    "language": lang,
                    "complexity_bucket": bucket,
                    "prediction_html": html,
                    "cell_bboxes": [],
                    "elapse_sec": elapse,
                    "status": "ok"
                }
            except Exception as e:
                rec = {
                    "image_id": img_id,
                    "language": lang,
                    "complexity_bucket": bucket,
                    "prediction_html": "<table><tbody></tbody></table>",
                    "cell_bboxes": [],
                    "elapse_sec": 0.0,
                    "status": "error",
                    "error": str(e)
                }
            records.append(rec)
            with open(out_file, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[UniTable] Finished {lang} ({len(records)} records).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--language", action="append", dest="languages", default=None)
    args = parser.parse_args()

    run_batch(args.data_root, args.output_dir, args.device, languages=args.languages)
