#!/usr/bin/env python3
"""Benchmark Evaluation Pipeline for Blank Cells Tables (500 samples):

Greedy GPU Saturation Scheduler:
- Fully saturates GPU 0's memory and compute utilization first (up to 6 concurrent models / 30 GB VRAM).
- Only when GPU 0 is fully occupied does it allocate remaining models to GPU 1.
- Leaves other GPUs completely untouched.
- Evaluates all 10 models:
  TATR v1.1-All, SLANet-plus, UniTable, TableFormerV2, LORE-TSR,
  Cycle-CenterNet, MTL-TabNet, PIX-TAB, TFLOP, SPRINT
- CPU affinity: taskset -c 16-47.
- Computes TEDS-Struct & GriTS-Top scores immediately upon inference completion.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

BASE_DIR = Path("/scratch/tanish/tanishq/table-understanding/repos/synthetic-table-generation")
SCRIPT_DIR = Path("/scratch/tanish/tanishq/table-understanding/repos/inference_scripts/batch_eval")
CONDA_BASE = Path("/home/tanish/scratch/miniconda3/envs")
METRIC_PYTHON = CONDA_BASE / "tables-detr" / "bin" / "python"
TOTAL_SAMPLES = 500

DATASET_CFG = {
    "key": "blank_cells",
    "name": "Blank Cells Tables (40-60% Blank - 500 Samples)",
    "data_root": BASE_DIR / "blank_cells_tables_500",
    "out_root": BASE_DIR / "eval_blank_cells_tables_500",
}

MODELS = [
    {"key": "tatr", "model_key": "tatr", "name": "TATR v1.1-All", "python": CONDA_BASE / "tables-detr" / "bin" / "python", "script": SCRIPT_DIR / "batch_tatr.py"},
    {"key": "slanet", "model_key": "slanet", "name": "SLANet-plus", "python": CONDA_BASE / "slanet" / "bin" / "python", "script": SCRIPT_DIR / "batch_slanet.py"},
    {"key": "unitable", "model_key": "unitable", "name": "UniTable", "python": CONDA_BASE / "unitable" / "bin" / "python", "script": SCRIPT_DIR / "batch_unitable.py"},
    {"key": "tableformer", "model_key": "tableformer", "name": "TableFormerV2", "python": CONDA_BASE / "tableformer" / "bin" / "python", "script": SCRIPT_DIR / "batch_tableformer.py"},
    {"key": "lore", "model_key": "lore", "name": "LORE-TSR", "python": CONDA_BASE / "lore-tsr" / "bin" / "python", "script": SCRIPT_DIR / "batch_lore.py"},
    {"key": "cycle_centernet", "model_key": "cycle_centernet", "name": "Cycle-CenterNet", "python": CONDA_BASE / "cycle-centernet" / "bin" / "python", "script": SCRIPT_DIR / "batch_cycle_centernet.py"},
    {"key": "mtltabnet", "model_key": "mtltabnet", "name": "MTL-TabNet", "python": CONDA_BASE / "mtltabnet" / "bin" / "python", "script": SCRIPT_DIR / "batch_mtltabnet.py"},
    {"key": "pixtab", "model_key": "pixtab", "name": "PIX-TAB", "python": CONDA_BASE / "mtltabnet" / "bin" / "python", "script": SCRIPT_DIR / "batch_pixtab.py"},
    {"key": "tflop", "model_key": "tflop", "name": "TFLOP", "python": CONDA_BASE / "tflop" / "bin" / "python", "script": SCRIPT_DIR / "batch_tflop.py"},
    {"key": "sprint", "model_key": "sprint", "name": "SPRINT", "python": CONDA_BASE / "sprint" / "bin" / "python", "script": SCRIPT_DIR / "batch_sprint.py"},
]

MODEL_VRAM_GB = {
    "tatr": 1.5,
    "slanet": 1.5,
    "unitable": 1.5,
    "tableformer": 2.5,
    "lore": 2.5,
    "cycle_centernet": 3.5,
    "tflop": 2.5,
    "sprint": 3.5,
    "pixtab": 9.5,
    "mtltabnet": 9.5,
}

# Ordered priority list of GPUs: GPU 0 is fully filled first; GPU 1 takes overflow
AVAILABLE_GPUS = [0, 1]
MAX_JOBS_PER_GPU = 6
MAX_VRAM_PER_GPU = 30.0  # GB


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def find_available_gpu(model_key: str, active_jobs_by_gpu: dict) -> int | None:
    req_vram = MODEL_VRAM_GB.get(model_key, 2.5)
    for gpu in AVAILABLE_GPUS:
        jobs = active_jobs_by_gpu.get(gpu, [])
        curr_vram = sum(MODEL_VRAM_GB.get(j["model_key"], 2.5) for j in jobs)
        if len(jobs) < MAX_JOBS_PER_GPU and (curr_vram + req_vram) <= MAX_VRAM_PER_GPU:
            return gpu
    return None


def print_dataset_leaderboard(dataset_cfg: dict) -> list:
    out_root = dataset_cfg["out_root"]
    leaderboard = []
    for m in MODELS:
        mkey = m["key"]
        summary_file = out_root / mkey / "summary.json"
        if not summary_file.exists():
            continue
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            ov = data.get("overall", {})
            langs = data.get("languages", {})
            chem_data = langs.get("chemical", ov)
            by_comp = chem_data.get("by_complexity", {})
            row = {
                "model_key": mkey,
                "model_name": m["name"],
                "total_records": data.get("total_records", 0),
                "valid": ov.get("valid", 0),
                "teds_struct": chem_data.get("teds_struct", 0.0),
                "grits_top": chem_data.get("grits_top", 0.0),
                "grits_precision": chem_data.get("grits_top_precision", 0.0),
                "grits_recall": chem_data.get("grits_top_recall", 0.0),
                "simple_teds": by_comp.get("simple", {}).get("teds_struct", 0.0),
                "complex_teds": by_comp.get("complex", {}).get("teds_struct", 0.0),
            }
            leaderboard.append(row)
        except Exception as e:
            print(f"Error reading summary for {mkey}: {e}")

    leaderboard.sort(key=lambda x: x["teds_struct"], reverse=True)
    out_file = out_root / "final_leaderboard.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    print("\n" + "=" * 95)
    print(f"LEADERBOARD: {dataset_cfg['name']}")
    print("=" * 95)
    print(f"{'Rank':<5} | {'Model Name':<18} | {'TEDS-Struct':<12} | {'GriTS-Top':<10} | {'GriTS P':<8} | {'GriTS R':<8} | {'Status'}")
    print("-" * 95)
    for i, r in enumerate(leaderboard, 1):
        print(f"{i:<5} | {r['model_name']:<18} | {r['teds_struct']*100:6.2f}%     | {r['grits_top']*100:6.2f}%   | {r['grits_precision']*100:5.2f}%  | {r['grits_recall']*100:5.2f}%  | Scored ({r['total_records']}/500)")
    print("=" * 95 + "\n")
    return leaderboard


def run_benchmark():
    data_root = DATASET_CFG["data_root"]
    out_root = DATASET_CFG["out_root"]
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 80)
    print(f"STARTING BENCHMARK: {DATASET_CFG['name']}")
    print(f"Data Root:   {data_root}")
    print(f"Output Root: {out_root}")
    print(f"Models:      {len(MODELS)}")
    print(f"GPU Policy:  Fully saturate GPU {AVAILABLE_GPUS[0]} first (up to {MAX_JOBS_PER_GPU} jobs / {MAX_VRAM_PER_GPU}GB), spill to GPU {AVAILABLE_GPUS[1]}")
    print("CPU affinity: taskset -c 16-47 (leaving cores 0-15 free)")
    print("#" * 80 + "\n")

    for m in MODELS:
        (out_root / m["key"] / "predictions").mkdir(parents=True, exist_ok=True)
        (out_root / m["key"] / "scores").mkdir(parents=True, exist_ok=True)

    # Active inference jobs: list of (proc, fp, job, gpu)
    active_inferences = []
    # Active metric jobs: dict of mkey: (proc, fp)
    active_metrics = {}
    completed_scoring = set()
    started_inference = set()

    start_time = time.time()
    last_status = 0

    while len(completed_scoring) < len(MODELS):
        time.sleep(3)
        now = time.time()

        # 1. Check completed inference processes
        still_active = []
        for proc, fp, job, gpu in active_inferences:
            poll = proc.poll()
            if poll is not None:
                fp.close()
                mkey = job["model_key"]
                lines = count_lines(out_root / mkey / "predictions" / "chemical.jsonl")
                print(f"[INFERENCE COMPLETE] {job['name']} on GPU {gpu} finished (code {poll}, {lines}/{TOTAL_SAMPLES} samples).")
            else:
                still_active.append((proc, fp, job, gpu))
        active_inferences = still_active

        # Build current GPU allocation map
        active_jobs_by_gpu = {g: [] for g in AVAILABLE_GPUS}
        for _, _, job, gpu in active_inferences:
            active_jobs_by_gpu[gpu].append(job)

        # 2. Greedily dispatch unstarted models (priority: pack GPU 0 completely first)
        for job in MODELS:
            mkey = job["model_key"]
            if mkey in started_inference:
                continue

            pred_file = out_root / mkey / "predictions" / "chemical.jsonl"
            if count_lines(pred_file) >= TOTAL_SAMPLES:
                started_inference.add(mkey)
                continue

            target_gpu = find_available_gpu(mkey, active_jobs_by_gpu)
            if target_gpu is not None:
                log_file = logs_dir / f"{mkey}_inference.log"
                fp = open(log_file, "a", buffering=1)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(target_gpu)
                env["PYTHONUNBUFFERED"] = "1"

                cmd = [
                    "taskset", "-c", "16-47",
                    str(job["python"]),
                    str(job["script"]),
                    "--data-root", str(data_root),
                    "--output-dir", str(out_root / mkey),
                    "--device", "cuda:0",
                    "--language", "chemical"
                ]

                req_vram = MODEL_VRAM_GB.get(mkey, 2.5)
                tot_vram = sum(MODEL_VRAM_GB.get(j["model_key"], 2.5) for j in active_jobs_by_gpu[target_gpu]) + req_vram
                print(f"[LAUNCH INFERENCE] {job['name']} -> GPU {target_gpu} (Jobs: {len(active_jobs_by_gpu[target_gpu])+1}/{MAX_JOBS_PER_GPU}, Estimated VRAM: {tot_vram:.1f}/{MAX_VRAM_PER_GPU}GB)")

                proc = subprocess.Popen(cmd, env=env, stdout=fp, stderr=subprocess.STDOUT)
                active_inferences.append((proc, fp, job, target_gpu))
                active_jobs_by_gpu[target_gpu].append(job)
                started_inference.add(mkey)

        # 3. Check if any completed inference is ready for metric evaluation
        for m in MODELS:
            mkey = m["key"]
            if mkey not in active_metrics and mkey not in completed_scoring:
                pred_file = out_root / mkey / "predictions" / "chemical.jsonl"
                lines = count_lines(pred_file)
                if lines >= TOTAL_SAMPLES:
                    print(f"\n>>> [LAUNCH METRICS] {m['name']} finished inference ({lines} samples). Computing TEDS-Struct & GriTS-Top...")
                    metric_log = logs_dir / f"{mkey}_metrics.log"
                    m_fp = open(metric_log, "a", buffering=1)
                    m_cmd = [
                        "taskset", "-c", "16-47",
                        str(METRIC_PYTHON),
                        str(BASE_DIR / "score_chemical_flexible.py"),
                        "--data-root", str(data_root),
                        "--output-dir", str(out_root / mkey),
                        "--model-key", mkey,
                        "--model-name", m["name"],
                        "--workers", "16"
                    ]
                    m_proc = subprocess.Popen(m_cmd, stdout=m_fp, stderr=subprocess.STDOUT)
                    active_metrics[mkey] = (m_proc, m_fp)

        # 4. Check active metric scoring processes
        for mkey, (proc, fp) in list(active_metrics.items()):
            poll = proc.poll()
            if poll is not None:
                fp.close()
                completed_scoring.add(mkey)
                del active_metrics[mkey]
                print(f"[METRICS COMPLETE] {mkey} scored successfully!")
                print_dataset_leaderboard(DATASET_CFG)

        # 5. Periodic status report every 30s
        if now - last_status >= 30:
            last_status = now
            elapsed = int(now - start_time)
            mins, secs = divmod(elapsed, 60)
            status_parts = []
            for m in MODELS:
                mkey = m["key"]
                lines = count_lines(out_root / mkey / "predictions" / "chemical.jsonl")
                summary_exists = (out_root / mkey / "summary.json").exists()
                if summary_exists:
                    status_parts.append(f"{mkey}:DONE")
                elif lines >= TOTAL_SAMPLES:
                    status_parts.append(f"{mkey}:SCORING")
                elif lines > 0:
                    status_parts.append(f"{mkey}:{lines}/500")
                else:
                    status_parts.append(f"{mkey}:WAIT")

            gpu0_jobs = len([j for _, _, j, g in active_inferences if g == 0])
            gpu1_jobs = len([j for _, _, j, g in active_inferences if g == 1])
            gpu0_vram = sum(MODEL_VRAM_GB.get(j["model_key"], 2.5) for _, _, j, g in active_inferences if g == 0)
            gpu1_vram = sum(MODEL_VRAM_GB.get(j["model_key"], 2.5) for _, _, j, g in active_inferences if g == 1)

            print(f"[{mins:02d}:{secs:02d}] Active: GPU 0: {gpu0_jobs} jobs ({gpu0_vram:.1f}GB) | GPU 1: {gpu1_jobs} jobs ({gpu1_vram:.1f}GB) || " + " | ".join(status_parts))

    print("\n" + "=" * 80)
    print("ALL MODELS COMPLETED!")
    print("=" * 80)
    print_dataset_leaderboard(DATASET_CFG)


if __name__ == "__main__":
    run_benchmark()
