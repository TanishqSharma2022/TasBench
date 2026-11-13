#!/usr/bin/env python3
"""Flexible Parallel Record-Level Evaluation for Chemical Tables.

Computes:
- TEDS-Struct (Tree Edit Distance on structure HTML via Apted)
- GriTS-Top (Grid Table Similarity on topological grids: F1, Precision, Recall)
Environment: tables-detr
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from statistics import mean
from concurrent.futures import ProcessPoolExecutor
from bs4 import BeautifulSoup

REPO_ROOT = Path("/home/tanish/scratch/tanishq/table-understanding/repos/table-transformer")
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT / "src"]
import grits
from grits import grits_top, html_to_grids

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.append(str(REPO_ROOT / "detr"))
from metric import TEDS

# Global per worker
worker_teds = None

def init_worker():
    global worker_teds
    worker_teds = TEDS(structure_only=True)

def clean_and_canonicalize(html_str: str) -> str:
    if not html_str:
        return "<table><tbody></tbody></table>"
    if "<table" not in html_str.lower():
        html_str = f"<table>{html_str}</table>"

    soup = BeautifulSoup(html_str, "html.parser")
    for colgroup in soup.find_all(["colgroup", "col"]):
        colgroup.decompose()
    for thead in soup.find_all("thead"):
        if not thead.find("tr"):
            tr = soup.new_tag("tr")
            for child in list(thead.children):
                tr.append(child)
            thead.append(tr)
    for th in soup.find_all("th"):
        th.name = "td"
    table = soup.find("table")
    if table:
        tbody = table.find("tbody")
        if not tbody:
            tbody = soup.new_tag("tbody")
            for child in list(table.children):
                if child.name == "tr":
                    tbody.append(child)
            table.append(tbody)
    return str(table)

def document_html(table_html: str) -> str:
    return f"<html><body>{table_html}</body></html>"

def score_item(item):
    ann, pred = item
    img_id = ann["image_id"]
    lang = "chemical"
    comp = ann.get("complexity_bucket", "simple")

    gt_html_raw = ann.get("ground_truth", {}).get("structure_html", "")
    if not gt_html_raw:
        tokens = ann.get("html", {}).get("structure", {}).get("tokens", [])
        gt_html_raw = "".join(tokens) if tokens else "<table><tbody></tbody></table>"
    pred_html_raw = pred.get("prediction_html", "<table><tbody></tbody></table>")

    gt_html = clean_and_canonicalize(gt_html_raw)
    pred_html = clean_and_canonicalize(pred_html_raw)

    # TEDS-Struct
    try:
        teds_val = worker_teds.evaluate(document_html(pred_html), document_html(gt_html))
        teds_score = max(0.0, min(1.0, float(teds_val)))
    except Exception:
        teds_score = 0.0

    # GriTS-Top
    try:
        pred_grids = html_to_grids(pred_html)
        gt_grids = html_to_grids(gt_html)
        if pred_grids is not None and gt_grids is not None:
            top_f1, top_p, top_r = grits_top(gt_grids["top"], pred_grids["top"])
            grits_val = float(top_f1)
            grits_p = float(top_p)
            grits_r = float(top_r)
        else:
            grits_val, grits_p, grits_r = 0.0, 0.0, 0.0
    except Exception:
        grits_val, grits_p, grits_r = 0.0, 0.0, 0.0

    return {
        "image_id": img_id,
        "language": lang,
        "complexity_bucket": comp,
        "status": "ok" if pred.get("status") == "ok" else "error",
        "teds_struct": teds_score,
        "grits_top": grits_val,
        "grits_top_precision": grits_p,
        "grits_top_recall": grits_r,
        "prediction_html": pred_html
    }

def evaluate_model(data_root: Path, output_dir: Path, model_key: str, model_name: str, num_workers: int = 16):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    pred_file = output_dir / "predictions" / "chemical.jsonl"
    
    ann_file = data_root / "annotations" / "chemical.jsonl"
    if not ann_file.exists():
        ann_file = data_root / "annotations.jsonl"

    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    if not pred_file.exists():
        print(f"[{model_key}] Prediction file not found: {pred_file}")
        return None

    with open(ann_file, "r", encoding="utf-8") as f:
        ann_records = [json.loads(line) for line in f if line.strip()]

    with open(pred_file, "r", encoding="utf-8") as f:
        pred_map = {json.loads(line)["image_id"]: json.loads(line) for line in f if line.strip()}

    items = [(ann, pred_map.get(ann["image_id"], {})) for ann in ann_records]
    print(f"[{model_name}] Starting parallel scoring of {len(items)} items using {num_workers} workers...")
    start_t = time.time()

    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        scored_records = list(executor.map(score_item, items, chunksize=5))

    elapsed = time.time() - start_t
    print(f"[{model_name}] Finished scoring in {elapsed:.1f}s!")

    score_file = scores_dir / "chemical.jsonl"
    with open(score_file, "w", encoding="utf-8") as f:
        for r in scored_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_by_lang = {
        "chemical": {
            "count": len(scored_records),
            "valid": sum(r["status"] == "ok" for r in scored_records),
            "teds_struct": mean([r["teds_struct"] for r in scored_records]) if scored_records else 0.0,
            "grits_top": mean([r["grits_top"] for r in scored_records]) if scored_records else 0.0,
            "grits_top_precision": mean([r["grits_top_precision"] for r in scored_records]) if scored_records else 0.0,
            "grits_top_recall": mean([r["grits_top_recall"] for r in scored_records]) if scored_records else 0.0,
            "by_complexity": {
                c: {
                    "count": len([r for r in scored_records if r["complexity_bucket"] == c]),
                    "teds_struct": mean([r["teds_struct"] for r in scored_records if r["complexity_bucket"] == c]) if any(r["complexity_bucket"] == c for r in scored_records) else 0.0,
                    "grits_top": mean([r["grits_top"] for r in scored_records if r["complexity_bucket"] == c]) if any(r["complexity_bucket"] == c for r in scored_records) else 0.0
                }
                for c in ["simple", "medium", "complex"]
            }
        }
    }

    overall_summary = {
        "model": model_name,
        "total_records": len(scored_records),
        "overall": {
            "count": len(scored_records),
            "valid": sum(r["status"] == "ok" for r in scored_records),
            "teds_struct": mean([r["teds_struct"] for r in scored_records]) if scored_records else 0.0,
            "grits_top": mean([r["grits_top"] for r in scored_records]) if scored_records else 0.0,
            "grits_top_precision": mean([r["grits_top_precision"] for r in scored_records]) if scored_records else 0.0,
            "grits_top_recall": mean([r["grits_top_recall"] for r in scored_records]) if scored_records else 0.0,
        },
        "overall_by_complexity": {
            c: {
                "count": len([r for r in scored_records if r["complexity_bucket"] == c]),
                "teds_struct": mean([r["teds_struct"] for r in scored_records if r["complexity_bucket"] == c]) if any(r["complexity_bucket"] == c for r in scored_records) else 0.0,
                "grits_top": mean([r["grits_top"] for r in scored_records if r["complexity_bucket"] == c]) if any(r["complexity_bucket"] == c for r in scored_records) else 0.0
            }
            for c in ["simple", "medium", "complex"]
        },
        "languages": summary_by_lang
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)

    print(f"[{model_name}] Summary saved. TEDS-Struct: {overall_summary['overall']['teds_struct']:.4f}, GriTS-Top: {overall_summary['overall']['grits_top']:.4f}")
    return overall_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-key", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    evaluate_model(args.data_root, args.output_dir, args.model_key, args.model_name, args.workers)
