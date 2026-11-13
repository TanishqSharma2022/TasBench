#!/usr/bin/env python3
"""
Batch Inference for TFLOP.
Environment: tflop
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from tqdm import tqdm

repo_dir = "/home/tanish/scratch/tanishq/table-understanding/repos/TFLOP"
sys.path.insert(0, repo_dir)
from tflop.model.model.TFLOP import TFLOP
from tflop.model.model.TFLOP_Config import TFLOPConfig

sys.path.insert(0, "/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts")
from otsl_utils import otsl_to_html

ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

def run_batch(data_root: Path, output_dir: Path, device: str = "cuda:0", languages=None):
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TFLOP] Loading model on {device}...")
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

    transform = transforms.Compose([
        transforms.Resize((config.input_size[0], config.input_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    target_languages = languages or ALL_LANGUAGES
    for lang in target_languages:
        ann_file = data_root / "annotations" / f"{lang}.jsonl"

        with open(ann_file, "r", encoding="utf-8") as f:

            items = [json.loads(line) for line in f if line.strip()]


        out_file = preds_dir / f"{lang}.jsonl"

        if out_file.exists():

            with open(out_file, "r") as f:

                existing = [line for line in f if line.strip()]

            if len(existing) >= len(items):

                print(f"[TFLOP] {lang} already completed ({len(existing)} records). Skipping.")

                continue

        records = []
        for item in tqdm(items, desc=f"TFLOP [{lang}]"):
            img_rel = item["image"]
            img_path = str(data_root / img_rel)
            img_id = item["image_id"]
            bucket = item.get("complexity_bucket", "unknown")

            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(dev).bfloat16()

                prompt_ids = tokenizer("<s><s_answer>", add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
                coords_int_padded = torch.full((1, config.bbox_token_cnt, 4), 771, dtype=torch.long, device=dev)
                pointer_args = {
                    "coord_input_idx": coords_int_padded,
                    "coord_input_length": torch.tensor([1], dtype=torch.long, device=dev)
                }

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

                rec = {
                    "image_id": img_id,
                    "language": lang,
                    "complexity_bucket": bucket,
                    "prediction_html": html_text,
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

        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[TFLOP] Finished {lang} ({len(records)} records).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--language", action="append", dest="languages", default=None)
    args = parser.parse_args()

    run_batch(args.data_root, args.output_dir, args.device, languages=args.languages)
