#!/usr/bin/env python3
"""Master Evaluation Pipeline for Dual Chemical Tables Benchmarks (500 samples each):

1. chemical_tables_smiles_500
2. chemical_tables_combined_500

Evaluates on all 10 models (SmolDocling excluded):
- TATR v1.1-All
- SLANet-plus
- UniTable
- TableFormerV2
- LORE-TSR
- Cycle-CenterNet
- MTL-TabNet
- PIX-TAB
- TFLOP
- SPRINT

Runs parallel inference queues across available NVIDIA A100 GPUs (GPUs 7, 2, 5, 6),
pins CPU tasks to cores 16-47 (leaving 0-15 untouched),
and computes TEDS-Struct & GriTS-Top scores immediately upon completion.
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

DATASETS = [
    {
        "key": "smiles",
        "name": "Chemical Tables (SMILES Only - 500 Samples)",
        "data_root": BASE_DIR / "chemical_tables_smiles_500",
        "out_root": BASE_DIR / "eval_chemical_tables_smiles_500",
    },
    {
        "key": "combined",
        "name": "Chemical Tables (Combined Images + SMILES - 500 Samples)",
        "data_root": BASE_DIR / "chemical_tables_combined_500",
        "out_root": BASE_DIR / "eval_chemical_tables_combined_500",
    },
]

# 10 Models (SmolDocling strictly excluded)
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

# GPU Allocation Map (Using GPUs 7, 2, 5, 6)
GPU_QUEUES = {
    7: [
        MODELS[0], # tatr
        MODELS[1], # slanet
        MODELS[5], # cycle_centernet
    ],
    2: [
        MODELS[2], # unitable
        MODELS[3], # tableformer
    ],
    5: [
        MODELS[4], # lore
        MODELS[6], # mtltabnet
        MODELS[7], # pixtab
    ],
    6: [
        MODELS[9], # sprint
        MODELS[8], # tflop
    ],
}


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


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


def run_single_dataset(dataset_cfg: dict):
    data_root = dataset_cfg["data_root"]
    out_root = dataset_cfg["out_root"]
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 80)
    print(f"STARTING BENCHMARK: {dataset_cfg['name']}")
    print(f"Data Root:   {data_root}")
    print(f"Output Root: {out_root}")
    print(f"Models:      {len(MODELS)} (SmolDocling excluded)")
    print("CPU affinity: taskset -c 16-47 (leaving cores 0-15 free)")
    print("#" * 80 + "\n")

    for m in MODELS:
        (out_root / m["key"] / "predictions").mkdir(parents=True, exist_ok=True)
        (out_root / m["key"] / "scores").mkdir(parents=True, exist_ok=True)

    active_gpu_procs = {} # gpu: (proc, fp, job)
    gpu_queue_idx = {gpu: 0 for gpu in GPU_QUEUES}
    active_metric_procs = {} # model_key: (proc, fp)
    completed_scoring = set()

    start_time = time.time()
    last_status = 0

    while len(completed_scoring) < len(MODELS):
        time.sleep(5)
        now = time.time()

        # 1. Check active GPU inference processes
        for gpu in list(active_gpu_procs.keys()):
            proc, fp, job = active_gpu_procs[gpu]
            poll = proc.poll()
            if poll is not None:
                fp.close()
                mkey = job["model_key"]
                lines = count_lines(out_root / mkey / "predictions" / "chemical.jsonl")
                print(f"[INFERENCE COMPLETE] {job['name']} on GPU {gpu} finished (code {poll}, {lines}/{TOTAL_SAMPLES} samples).")
                del active_gpu_procs[gpu]

        # 2. Launch next inference job in idle GPU queues
        for gpu, jobs in GPU_QUEUES.items():
            if gpu not in active_gpu_procs:
                idx = gpu_queue_idx[gpu]
                while idx < len(jobs):
                    cand_job = jobs[idx]
                    mkey = cand_job["model_key"]
                    pred_file = out_root / mkey / "predictions" / "chemical.jsonl"
                    if count_lines(pred_file) >= TOTAL_SAMPLES:
                        idx += 1
                        gpu_queue_idx[gpu] = idx
                        continue
                    break

                if idx < len(jobs):
                    job = jobs[idx]
                    mkey = job["model_key"]
                    log_file = logs_dir / f"{mkey}_inference.log"
                    fp = open(log_file, "a", buffering=1)
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
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
                    print(f"[LAUNCH INFERENCE] {job['name']} -> GPU {gpu} (taskset 16-47)")
                    proc = subprocess.Popen(cmd, env=env, stdout=fp, stderr=subprocess.STDOUT)
                    active_gpu_procs[gpu] = (proc, fp, job)
                    gpu_queue_idx[gpu] = idx + 1

        # 3. Check if any model's predictions are complete and launch metric evaluation
        for m in MODELS:
            mkey = m["key"]
            if mkey not in active_metric_procs and mkey not in completed_scoring:
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
                    active_metric_procs[mkey] = (m_proc, m_fp)

        # 4. Check active metric scoring processes
        for mkey, (proc, fp) in list(active_metric_procs.items()):
            poll = proc.poll()
            if poll is not None:
                fp.close()
                completed_scoring.add(mkey)
                del active_metric_procs[mkey]
                print(f"[METRICS COMPLETE] {mkey} scored successfully!")
                print_dataset_leaderboard(dataset_cfg)

        # 5. Periodic status report every 30s
        if now - last_status >= 30:
            last_status = now
            elapsed = int(now - start_time)
            mins, secs = divmod(elapsed, 60)
            print(f"--- [{dataset_cfg['key'].upper()}] Progress: {mins:02d}m {secs:02d}s elapsed | {len(completed_scoring)}/{len(MODELS)} models scored ---")
            for m in MODELS:
                mkey = m["key"]
                lines = count_lines(out_root / mkey / "predictions" / "chemical.jsonl")
                if mkey in completed_scoring:
                    st = "DONE (Scored)"
                elif mkey in active_metric_procs:
                    st = "Scoring..."
                elif lines >= TOTAL_SAMPLES:
                    st = "Inference Complete (Queued for Scoring)"
                elif mkey in [job["model_key"] for proc, fp, job in active_gpu_procs.values()]:
                    st = f"Inferring ({lines}/{TOTAL_SAMPLES})"
                else:
                    st = "Queued"
                print(f"  * {m['name']:<18}: {st}")

    print(f"\nALL 10 MODELS FINISHED FOR: {dataset_cfg['name']}!")
    print_dataset_leaderboard(dataset_cfg)


def main():
    print("=" * 80)
    print("DUAL CHEMICAL BENCHMARK MASTER EVALUATION RUNNER")
    print(f"Target Datasets: {len(DATASETS)}")
    print(f"Target Models:   {len(MODELS)} (excluding SmolDocling)")
    print("=" * 80)

    for ds in DATASETS:
        run_single_dataset(ds)

    print("\n" + "=" * 80)
    print("ALL DUAL BENCHMARKS FULLY COMPLETED AND SCORED!")
    print("=" * 80)
    for ds in DATASETS:
        print_dataset_leaderboard(ds)


if __name__ == "__main__":
    main()
