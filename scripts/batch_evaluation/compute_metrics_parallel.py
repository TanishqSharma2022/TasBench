#!/usr/bin/env python3
"""
Parallel Standardized Metric Evaluation Engine for TSR Models on Multilingual Table Benchmark.
Computes:
- TEDS-Struct (Tree Edit Distance on structure HTML via Apted)
- GriTS-Top (Grid Table Similarity on topological grids: F1, Precision, Recall)
Outputs per-language scored jsonl files and comprehensive summary.json.
Uses multiprocessing across languages for fast throughput.
Environment: tables-detr
"""

import os
import sys
import json
import argparse
from pathlib import Path
from statistics import mean
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from bs4 import BeautifulSoup
from tqdm import tqdm

# Ensure unshadowed grits import
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path("/home/tanish/scratch/tanishq/table-understanding/repos/table-transformer")
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT / "src"]

import grits
from grits import grits_top, html_to_grids

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.append(str(REPO_ROOT / "detr"))
from metric import TEDS

ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

def clean_and_canonicalize(html_str: str) -> str:
    """Canonicalize HTML for TEDS and GriTS parsers to prevent syntax crashes."""
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

def evaluate_single_language(pred_file: Path, ann_file: Path, lang: str, model_name: str):
    """Worker function to evaluate a single language."""
    if not pred_file.exists():
        print(f"[WARN] Prediction file missing for {lang}: {pred_file}")
        return lang, []

    with open(pred_file, "r", encoding="utf-8") as f:
        pred_records = {json.loads(line)["image_id"]: json.loads(line) for line in f if line.strip()}

    with open(ann_file, "r", encoding="utf-8") as f:
        ann_records = [json.loads(line) for line in f if line.strip()]

    teds_metric = TEDS(structure_only=True)
    lang_scored = []

    for ann in tqdm(ann_records, desc=f"Scoring {model_name} [{lang}]", position=ALL_LANGUAGES.index(lang) if lang in ALL_LANGUAGES else 0):
        img_id = ann["image_id"]
        pred = pred_records.get(img_id, {})

        gt_html_raw = ann.get("ground_truth", {}).get("structure_html", "<table><tbody></tbody></table>")
        pred_html_raw = pred.get("prediction_html", "<table><tbody></tbody></table>")

        gt_html = clean_and_canonicalize(gt_html_raw)
        pred_html = clean_and_canonicalize(pred_html_raw)

        # Compute TEDS-Struct
        try:
            teds_val = teds_metric.evaluate(document_html(pred_html), document_html(gt_html))
            teds_score = max(0.0, min(1.0, float(teds_val)))
        except Exception:
            teds_score = 0.0

        # Compute GriTS-Top
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

        rec = {
            "image_id": img_id,
            "language": lang,
            "complexity_bucket": ann.get("complexity_bucket", "simple"),
            "status": "ok" if pred.get("status") == "ok" else "error",
            "teds_struct": teds_score,
            "grits_top": grits_val,
            "grits_top_precision": grits_p,
            "grits_top_recall": grits_r,
            "prediction_html": pred_html
        }
        lang_scored.append(rec)

    return lang, lang_scored

def evaluate_predictions_parallel(pred_dir: Path, data_root: Path, output_dir: Path, model_name: str = "Model", max_workers: int = 11, languages: list = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    summary_by_lang = {}
    all_scored_records = []

    if languages:
        target_langs = languages
    else:
        pred_langs = [f.stem for f in pred_dir.glob("*.jsonl")]
        target_langs = [l for l in ALL_LANGUAGES if l in pred_langs] if any(l in pred_langs for l in ALL_LANGUAGES) else pred_langs

    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for lang in target_langs:
            pred_file = pred_dir / f"{lang}.jsonl"
            ann_file = data_root / "annotations" / f"{lang}.jsonl"
            tasks.append(executor.submit(evaluate_single_language, pred_file, ann_file, lang, model_name))

        for future in as_completed(tasks):
            lang, lang_scored = future.result()
            if not lang_scored:
                continue

            # Save per-language scores
            with open(scores_dir / f"{lang}.jsonl", "w", encoding="utf-8") as f:
                for r in lang_scored:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            all_scored_records.extend(lang_scored)

            summary_by_lang[lang] = {
                "count": len(lang_scored),
                "valid": sum(r["status"] == "ok" for r in lang_scored),
                "teds_struct": mean([r["teds_struct"] for r in lang_scored]) if lang_scored else 0.0,
                "grits_top": mean([r["grits_top"] for r in lang_scored]) if lang_scored else 0.0,
                "grits_top_precision": mean([r["grits_top_precision"] for r in lang_scored]) if lang_scored else 0.0,
                "grits_top_recall": mean([r["grits_top_recall"] for r in lang_scored]) if lang_scored else 0.0,
                "by_complexity": {
                    c: {
                        "count": len([r for r in lang_scored if r["complexity_bucket"] == c]),
                        "teds_struct": mean([r["teds_struct"] for r in lang_scored if r["complexity_bucket"] == c]) if any(r["complexity_bucket"] == c for r in lang_scored) else 0.0,
                        "grits_top": mean([r["grits_top"] for r in lang_scored if r["complexity_bucket"] == c]) if any(r["complexity_bucket"] == c for r in lang_scored) else 0.0
                    }
                    for c in ["simple", "complex"]
                }
            }
            print(f"[FINISHED] {lang}: TEDS-Struct = {summary_by_lang[lang]['teds_struct']:.4f}, GriTS-Top = {summary_by_lang[lang]['grits_top']:.4f}")

    # Overall summary
    overall_summary = {
        "model": model_name,
        "total_records": len(all_scored_records),
        "overall": {
            "count": len(all_scored_records),
            "valid": sum(r["status"] == "ok" for r in all_scored_records),
            "teds_struct": mean([r["teds_struct"] for r in all_scored_records]) if all_scored_records else 0.0,
            "grits_top": mean([r["grits_top"] for r in all_scored_records]) if all_scored_records else 0.0,
            "grits_top_precision": mean([r["grits_top_precision"] for r in all_scored_records]) if all_scored_records else 0.0,
            "grits_top_recall": mean([r["grits_top_recall"] for r in all_scored_records]) if all_scored_records else 0.0,
        },
        "overall_by_complexity": {
            c: {
                "count": len([r for r in all_scored_records if r["complexity_bucket"] == c]),
                "teds_struct": mean([r["teds_struct"] for r in all_scored_records if r["complexity_bucket"] == c]) if any(r["complexity_bucket"] == c for r in all_scored_records) else 0.0,
                "grits_top": mean([r["grits_top"] for r in all_scored_records if r["complexity_bucket"] == c]) if any(r["complexity_bucket"] == c for r in all_scored_records) else 0.0
            }
            for c in ["simple", "complex"]
        },
        "languages": summary_by_lang
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] {model_name} - Overall TEDS-Struct: {overall_summary['overall']['teds_struct']:.4f} | GriTS-Top: {overall_summary['overall']['grits_top']:.4f}")
    return overall_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="Model")
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--languages", nargs="*", default=None, help="Specific language keys to evaluate")
    args = parser.parse_args()

    evaluate_predictions_parallel(args.pred_dir, args.data_root, args.output_dir, args.model_name, args.workers, languages=args.languages)
