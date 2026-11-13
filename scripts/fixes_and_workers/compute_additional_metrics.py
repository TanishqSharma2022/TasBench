#!/usr/bin/env python3
"""
Compute TEDS-Struct-100 and Cell-IOU AP across all models and languages
on eval_multilingual_500_master_run.

Outputs updated summary.json per model, updated final_leaderboard.json,
and a comprehensive report.
"""

import os
import sys
import json
import time
from pathlib import Path
from statistics import mean
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import numpy as np

DATA_ROOT = Path("/scratch/tanish/tanishq/table-understanding/data/Table Datasets/pubtabnet/multilingual_500_master_run")
EVAL_ROOT = Path("/scratch/tanish/tanishq/table-understanding/repos/synthetic-table-generation/eval_multilingual_500_master_run")

ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

ALL_MODELS = [
    "unitable", "tatr", "slanet", "sprint", "mtltabnet",
    "tableformer", "lore", "tflop", "pixtab", "smoldocling", "cycle_centernet"
]

BBOX_MODELS = ["tatr", "slanet", "cycle_centernet", "pixtab", "tableformer", "mtltabnet", "lore", "unitable"]

MODEL_DISPLAY = {
    "unitable": "UniTable",
    "tatr": "TATR v1.1-All",
    "slanet": "SLANet-plus",
    "sprint": "SPRINT",
    "mtltabnet": "MTL-TabNet",
    "tableformer": "TableFormerV2",
    "lore": "LORE-TSR",
    "tflop": "TFLOP",
    "pixtab": "PIX-TAB",
    "smoldocling": "SmolDocling-256M",
    "cycle_centernet": "Cycle-CenterNet"
}

def load_gt_for_lang(lang: str):
    ann_file = DATA_ROOT / "annotations" / f"{lang}.jsonl"
    gt_map = {}
    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            img_id = item["image_id"]
            boxes = [c["rendered_bbox_xyxy_px"] for c in item["ground_truth"]["cells"] if "rendered_bbox_xyxy_px" in c]
            img_rel = item["image"]
            img_path = DATA_ROOT / img_rel
            try:
                with Image.open(img_path) as im:
                    im_w, im_h = im.size
            except Exception:
                im_w = max((b[2] for b in boxes), default=1000)
                im_h = max((b[3] for b in boxes), default=1000)
            gt_map[img_id] = {
                "boxes": np.array(boxes, dtype=np.float32),
                "size": (im_w, im_h),
                "complexity": item.get("complexity_bucket", "simple")
            }
    return gt_map

def extract_pred_boxes(model_name: str, item: dict, img_size: tuple):
    raw_boxes = item.get("cell_bboxes") or []
    if not raw_boxes:
        return np.zeros((0, 4), dtype=np.float32)
    im_w, im_h = img_size
    parsed = []
    if model_name == "pixtab":
        for b in raw_boxes:
            box = b["bbox"] if isinstance(b, dict) and "bbox" in b else b
            if len(box) == 4:
                parsed.append([min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])])
    elif model_name in ("lore", "slanet"):
        for b in raw_boxes:
            if len(b) >= 8:
                xs = b[0::2]
                ys = b[1::2]
                parsed.append([min(xs), min(ys), max(xs), max(ys)])
            elif len(b) == 4:
                parsed.append([min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])])
    elif model_name == "tableformer":
        for b in raw_boxes:
            if len(b) == 4:
                parsed.append([min(b[0], b[2]) * im_w, min(b[1], b[3]) * im_h, max(b[0], b[2]) * im_w, max(b[1], b[3]) * im_h])
    else:  # tatr, mtltabnet, cycle_centernet
        for b in raw_boxes:
            if len(b) == 4:
                parsed.append([min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])])
    return np.array(parsed, dtype=np.float32)

def eval_cell_iou(model_name: str, pred_items: list, gt_map: dict):
    if model_name not in BBOX_MODELS:
        return None

    iou_thrs = np.arange(0.50, 1.00, 0.05)
    total_gt = sum(len(gt_map[item["image_id"]]["boxes"]) for item in pred_items if item["image_id"] in gt_map)
    img_data = []

    for item in pred_items:
        img_id = item["image_id"]
        if img_id not in gt_map:
            continue
        gts = gt_map[img_id]["boxes"]
        im_size = gt_map[img_id]["size"]
        preds = extract_pred_boxes(model_name, item, im_size)

        if len(preds) == 0 or len(gts) == 0:
            img_data.append((len(preds), len(gts), None))
            continue

        x1 = np.maximum(preds[:, None, 0], gts[None, :, 0])
        y1 = np.maximum(preds[:, None, 1], gts[None, :, 1])
        x2 = np.minimum(preds[:, None, 2], gts[None, :, 2])
        y2 = np.minimum(preds[:, None, 3], gts[None, :, 3])
        w = np.maximum(0.0, x2 - x1)
        h = np.maximum(0.0, y2 - y1)
        inter = w * h
        area_p = np.maximum(0.0, preds[:, 2] - preds[:, 0]) * np.maximum(0.0, preds[:, 3] - preds[:, 1])
        area_g = np.maximum(0.0, gts[:, 2] - gts[:, 0]) * np.maximum(0.0, gts[:, 3] - gts[:, 1])
        union = area_p[:, None] + area_g[None, :] - inter
        iou_mat = np.where(union > 0, inter / union, 0.0)
        img_data.append((len(preds), len(gts), iou_mat))

    ap_by_thr = {}
    metrics_50 = {}
    recall_pts = np.linspace(0, 1, 101)

    for thr in iou_thrs:
        thr_round = round(float(thr), 2)
        tp_total = 0
        fp_total = 0

        for n_preds, n_gts, iou_mat in img_data:
            if n_preds == 0:
                continue
            if n_gts == 0 or iou_mat is None:
                fp_total += n_preds
                continue

            pairs = []
            for p_i in range(n_preds):
                for g_i in range(n_gts):
                    if iou_mat[p_i, g_i] >= thr:
                        pairs.append((iou_mat[p_i, g_i], p_i, g_i))
            pairs.sort(reverse=True, key=lambda x: x[0])

            used_p = set()
            used_g = set()
            for score, p_i, g_i in pairs:
                if p_i not in used_p and g_i not in used_g:
                    used_p.add(p_i)
                    used_g.add(g_i)

            tp = len(used_p)
            fp = n_preds - tp
            tp_total += tp
            fp_total += fp

        prec = tp_total / max(1, tp_total + fp_total)
        rec = tp_total / max(1, total_gt)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        ap = float(np.mean([prec if t <= rec else 0.0 for t in recall_pts]))
        ap_by_thr[thr_round] = ap
        if thr_round == 0.50:
            metrics_50 = {"precision": prec, "recall": rec, "f1": f1, "ap50": ap}

    map_50_95 = float(np.mean(list(ap_by_thr.values())))
    ap_50 = ap_by_thr[0.50]
    ap_75 = ap_by_thr[0.75]

    return {
        "ap50": ap_50,
        "ap75": ap_75,
        "map": map_50_95,
        "precision_50": metrics_50["precision"],
        "recall_50": metrics_50["recall"],
        "f1_50": metrics_50["f1"]
    }

def process_model_lang(args):
    model_name, lang, gt_map = args
    scores_file = EVAL_ROOT / model_name / "scores" / f"{lang}.jsonl"
    preds_file = EVAL_ROOT / model_name / "predictions" / f"{lang}.jsonl"

    if not scores_file.exists():
        return model_name, lang, None

    # Load existing scores
    with open(scores_file, "r", encoding="utf-8") as f:
        scored_records = [json.loads(line) for line in f if line.strip()]

    total_tables = len(scored_records)
    perfect_struct_count = sum(1 for r in scored_records if abs(r["teds_struct"] - 1.0) < 1e-5)
    perfect_grits_count = sum(1 for r in scored_records if abs(r.get("grits_top", 0) - 1.0) < 1e-5)

    teds_struct_100 = perfect_struct_count / max(1, total_tables)
    grits_top_100 = perfect_grits_count / max(1, total_tables)
    avg_teds_struct = mean(r["teds_struct"] for r in scored_records) if scored_records else 0.0
    avg_grits_top = mean(r["grits_top"] for r in scored_records) if scored_records else 0.0

    # Cell-IOU AP
    cell_iou_res = None
    if model_name in BBOX_MODELS and preds_file.exists():
        with open(preds_file, "r", encoding="utf-8") as f:
            pred_items = [json.loads(line) for line in f if line.strip()]
        cell_iou_res = eval_cell_iou(model_name, pred_items, gt_map)

    return model_name, lang, {
        "count": total_tables,
        "teds_struct": avg_teds_struct,
        "teds_struct_100": teds_struct_100,
        "grits_top": avg_grits_top,
        "grits_top_100": grits_top_100,
        "perfect_teds_count": perfect_struct_count,
        "perfect_grits_count": perfect_grits_count,
        "cell_iou": cell_iou_res
    }

def main():
    print("=" * 80)
    print("COMPUTING TEDS-Struct-100 AND Cell-IOU AP (Multilingual 500 Run)")
    print("=" * 80)

    # 1. Preload GT for all 11 languages
    print("Preloading GT annotations for all 11 languages...")
    gt_cache = {}
    for lang in ALL_LANGUAGES:
        t0 = time.time()
        gt_cache[lang] = load_gt_for_lang(lang)
        print(f"  * {lang:<12}: {len(gt_cache[lang])} tables loaded ({time.time() - t0:.2f}s)")

    # 2. Build tasks
    tasks = []
    for m in ALL_MODELS:
        for lang in ALL_LANGUAGES:
            tasks.append((m, lang, gt_cache[lang]))

    print(f"\nProcessing {len(tasks)} (model, language) evaluation units in parallel...")
    t_start = time.time()
    results = {m: {} for m in ALL_MODELS}

    with ProcessPoolExecutor(max_workers=24) as executor:
        for m, lang, res in executor.map(process_model_lang, tasks):
            if res is not None:
                results[m][lang] = res

    print(f"Completed metric calculations in {time.time() - t_start:.2f}s!\n")

    # 3. Aggregate per-model summary
    model_summaries = {}
    for m in ALL_MODELS:
        m_langs = results[m]
        if not m_langs:
            continue
        total_tables = sum(l["count"] for l in m_langs.values())
        avg_teds = mean(l["teds_struct"] for l in m_langs.values())
        avg_teds_100 = mean(l["teds_struct_100"] for l in m_langs.values())
        total_perfect_teds = sum(l["perfect_teds_count"] for l in m_langs.values())
        avg_grits = mean(l["grits_top"] for l in m_langs.values())
        avg_grits_100 = mean(l["grits_top_100"] for l in m_langs.values())
        total_perfect_grits = sum(l["perfect_grits_count"] for l in m_langs.values())

        # Cell-IOU metrics aggregate across languages
        bbox_metrics = None
        if m in BBOX_MODELS:
            has_box = [l["cell_iou"] for l in m_langs.values() if l.get("cell_iou") is not None]
            if has_box:
                bbox_metrics = {
                    "ap50": mean(b["ap50"] for b in has_box),
                    "ap75": mean(b["ap75"] for b in has_box),
                    "map": mean(b["map"] for b in has_box),
                    "precision_50": mean(b["precision_50"] for b in has_box),
                    "recall_50": mean(b["recall_50"] for b in has_box),
                    "f1_50": mean(b["f1_50"] for b in has_box),
                }

        model_summaries[m] = {
            "model_key": m,
            "model_name": MODEL_DISPLAY.get(m, m),
            "num_languages": len(m_langs),
            "total_tables": total_tables,
            "teds_struct": avg_teds,
            "teds_struct_100": avg_teds_100,
            "perfect_teds_count": total_perfect_teds,
            "grits_top": avg_grits,
            "grits_top_100": avg_grits_100,
            "perfect_grits_count": total_perfect_grits,
            "has_bboxes": (m in BBOX_MODELS),
            "cell_iou_metrics": bbox_metrics,
            "per_language": m_langs
        }

        # Update model's summary.json with the new metrics
        model_summary_file = EVAL_ROOT / m / "summary.json"
        if model_summary_file.exists():
            try:
                with open(model_summary_file, "r", encoding="utf-8") as f:
                    orig_summary = json.load(f)
                orig_summary["overall"]["teds_struct_100"] = avg_teds_100
                orig_summary["overall"]["grits_top_100"] = avg_grits_100
                if bbox_metrics:
                    orig_summary["overall"]["cell_iou"] = bbox_metrics
                for l, l_data in m_langs.items():
                    if l in orig_summary.get("languages", {}):
                        orig_summary["languages"][l]["teds_struct_100"] = l_data["teds_struct_100"]
                        orig_summary["languages"][l]["grits_top_100"] = l_data["grits_top_100"]
                        if l_data.get("cell_iou"):
                            orig_summary["languages"][l]["cell_iou"] = l_data["cell_iou"]
                with open(model_summary_file, "w", encoding="utf-8") as f:
                    json.dump(orig_summary, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error updating {model_summary_file}: {e}")

    # 4. Construct comprehensive leaderboard
    leaderboard = list(model_summaries.values())
    leaderboard.sort(key=lambda x: x["teds_struct"], reverse=True)

    leaderboard_file = EVAL_ROOT / "final_leaderboard_extended.json"
    with open(leaderboard_file, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)

    # Also update final_leaderboard.json
    final_lb_file = EVAL_ROOT / "final_leaderboard.json"
    try:
        with open(final_lb_file, "r", encoding="utf-8") as f:
            curr_lb = json.load(f)
        for row in curr_lb:
            mkey = row.get("model_key")
            if mkey in model_summaries:
                s = model_summaries[mkey]
                row["teds_struct_100"] = s["teds_struct_100"]
                row["grits_top_100"] = s["grits_top_100"]
                row["perfect_teds_count"] = s["perfect_teds_count"]
                row["cell_iou_ap50"] = s["cell_iou_metrics"]["ap50"] if s["cell_iou_metrics"] else None
                row["cell_iou_ap75"] = s["cell_iou_metrics"]["ap75"] if s["cell_iou_metrics"] else None
                row["cell_iou_map"] = s["cell_iou_metrics"]["map"] if s["cell_iou_metrics"] else None
                row["cell_iou_f1"] = s["cell_iou_metrics"]["f1_50"] if s["cell_iou_metrics"] else None
                for l in ALL_LANGUAGES:
                    if l in row.get("per_language", {}) and l in s["per_language"]:
                        l_src = s["per_language"][l]
                        row["per_language"][l]["teds_struct_100"] = l_src["teds_struct_100"]
                        row["per_language"][l]["grits_top_100"] = l_src["grits_top_100"]
                        if l_src.get("cell_iou"):
                            row["per_language"][l]["cell_iou_ap50"] = l_src["cell_iou"]["ap50"]
                            row["per_language"][l]["cell_iou_ap75"] = l_src["cell_iou"]["ap75"]
                            row["per_language"][l]["cell_iou_map"] = l_src["cell_iou"]["map"]
                            row["per_language"][l]["cell_iou_f1"] = l_src["cell_iou"]["f1_50"]
        with open(final_lb_file, "w", encoding="utf-8") as f:
            json.dump(curr_lb, f, indent=2, ensure_ascii=False)
        print(f"Updated {final_lb_file} successfully.")
    except Exception as e:
        print(f"Error updating final_leaderboard.json: {e}")

    # Print Overall Table
    print("\n" + "=" * 125)
    print(f"{'Rank':<5} | {'Model Name':<18} | {'TEDS-Struct':<12} | {'TEDS-100%':<12} | {'GriTS-Top':<11} | {'GriTS-100%':<11} | {'Cell AP50':<11} | {'Cell AP75':<11} | {'Cell mAP':<11}")
    print("=" * 125)
    for i, row in enumerate(leaderboard, 1):
        m = row["model_key"]
        teds = f"{row['teds_struct']*100:6.2f}%"
        teds_100 = f"{row['teds_struct_100']*100:6.2f}%"
        grits = f"{row['grits_top']*100:6.2f}%"
        grits_100 = f"{row['grits_top_100']*100:6.2f}%"
        if row["cell_iou_metrics"]:
            ap50 = f"{row['cell_iou_metrics']['ap50']*100:6.2f}%"
            ap75 = f"{row['cell_iou_metrics']['ap75']*100:6.2f}%"
            cmap = f"{row['cell_iou_metrics']['map']*100:6.2f}%"
        else:
            ap50 = "N/A (no bbox)"
            ap75 = "N/A (no bbox)"
            cmap = "N/A (no bbox)"
        print(f"{i:<5} | {row['model_name']:<18} | {teds:<12} | {teds_100:<12} | {grits:<11} | {grits_100:<11} | {ap50:<11} | {ap75:<11} | {cmap:<11}")
    print("=" * 125)

if __name__ == "__main__":
    main()
