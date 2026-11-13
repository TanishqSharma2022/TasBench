#!/usr/bin/env python3
"""
Master Parallel TSR Evaluation Pipeline on Hardest 200 Multilingual Synthetic Dataset.
Packs multiple models per GPU to saturate VRAM and compute before spilling over.
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path

DATA_ROOT = Path("/home/tanish/scratch/tanishq/table-understanding/data/hardest_200_multilingual_synthetic")
OUT_ROOT = Path("/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts/eval_hardest_200_multilingual_synthetic")
SCRIPT_DIR = Path("/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts/batch_eval")
CONDA_BASE = Path("/home/tanish/scratch/miniconda3/envs")
METRIC_PYTHON = CONDA_BASE / "tables-detr" / "bin" / "python"

LOGS_DIR = OUT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    # GPU 0: 3 light models (~18 GB VRAM, 85-95% compute util)
    {
        "name": "slanet",
        "display_name": "SLANet-plus",
        "gpu": 0,
        "python": CONDA_BASE / "slanet" / "bin" / "python",
        "script": SCRIPT_DIR / "batch_slanet.py"
    },
    {
        "name": "tatr",
        "display_name": "TATR v1.1-All",
        "gpu": 0,
        "python": CONDA_BASE / "tables-detr" / "bin" / "python",
        "script": SCRIPT_DIR / "batch_tatr.py"
    },
    {
        "name": "lore",
        "display_name": "LORE-TSR",
        "gpu": 0,
        "python": CONDA_BASE / "lore-tsr" / "bin" / "python",
        "script": SCRIPT_DIR / "batch_lore.py"
    },
    # GPU 1: 3 medium models (~20 GB VRAM, 85-95% compute util)
    {
        "name": "cycle_centernet",
        "display_name": "Cycle-CenterNet",
        "gpu": 1,
        "python": CONDA_BASE / "cycle-centernet" / "bin" / "python",
        "script": SCRIPT_DIR / "batch_cycle_centernet.py"
    },
    {
        "name": "sprint",
        "display_name": "SPRINT",
        "gpu": 1,
        "python": CONDA_BASE / "sprint" / "bin" / "python",
        "script": SCRIPT_DIR / "batch_sprint.py"
    },
    {
        "name": "tableformer",
        "display_name": "TableFormerV2",
        "gpu": 1,
        "python": CONDA_BASE / "tableformer" / "bin" / "python",
        "script": SCRIPT_DIR / "batch_tableformer.py"
    },
    # GPU 2: 3 medium/heavier models (~22 GB VRAM, 85-95% compute util)
    {
        "name": "tflop",
        "display_name": "TFLOP",
        "gpu": 2,
        "python": CONDA_BASE / "tflop" / "bin" / "python",
        "script": SCRIPT_DIR / "batch_tflop.py"
    },
    {
        "name": "pixtab",
        "display_name": "PIX-TAB",
        "gpu": 2,
        "python": CONDA_BASE / "mtltabnet" / "bin" / "python",
        "script": SCRIPT_DIR / "batch_pixtab.py"
    },
    {
        "name": "mtltabnet",
        "display_name": "MTL-TabNet",
        "gpu": 2,
        "python": CONDA_BASE / "mtltabnet" / "bin" / "python",
        "script": SCRIPT_DIR / "batch_mtltabnet.py"
    },
    # GPU 3: 2 heavier AR/VLM models (~22 GB VRAM, 80-90% compute util)
    {
        "name": "unitable",
        "display_name": "UniTable",
        "gpu": 3,
        "python": CONDA_BASE / "unitable" / "bin" / "python",
        "script": SCRIPT_DIR / "batch_unitable.py"
    },
    {
        "name": "smoldocling",
        "display_name": "SmolDocling-256M",
        "gpu": 3,
        "python": CONDA_BASE / "smoldocling" / "bin" / "python",
        "script": SCRIPT_DIR / "batch_smoldocling_sharded.py"
    }
]

def check_progress(model_name):
    preds_dir = OUT_ROOT / model_name / "predictions"
    if not preds_dir.exists():
        return 0, 0
    files = list(preds_dir.glob("*.jsonl"))
    completed_langs = 0
    total_samples = 0
    for f in files:
        try:
            with open(f, "r") as fp:
                lines = sum(1 for line in fp if line.strip())
            total_samples += lines
            if lines >= 200:
                completed_langs += 1
        except Exception:
            pass
    return completed_langs, total_samples

def get_gpu_stats():
    try:
        cmd = ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = res.stdout.strip().split("\n")
        stats = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                stats.append((parts[0], parts[1], parts[2], parts[3]))
        return stats
    except Exception:
        return []

def main():
    print("=" * 80)
    print("LAUNCHING MULTILINGUAL TSR BENCHMARK ON HARDEST 200 SYNTHETIC DATASET")
    print(f"Dataset root: {DATA_ROOT}")
    print(f"Output root:  {OUT_ROOT}")
    print("=" * 80)

    inference_procs = {}
    metric_procs = {}
    completed_inference = set()
    completed_metrics = set()

    # Launch inference for all models
    for m in MODELS:
        name = m["name"]
        model_out = OUT_ROOT / name
        model_out.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / f"{name}_inference.log"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(m["gpu"])
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [
            str(m["python"]),
            str(m["script"]),
            "--data-root", str(DATA_ROOT),
            "--output-dir", str(model_out),
            "--device", "cuda:0"
        ]

        print(f"[LAUNCH] {m['display_name']} on physical GPU {m['gpu']} -> logging to {log_file.name}")
        fp = open(log_file, "a")
        proc = subprocess.Popen(cmd, env=env, stdout=fp, stderr=subprocess.STDOUT)
        inference_procs[name] = (proc, fp, m)

    print("\nAll 11 models launched. Monitoring progress...\n")

    start_time = time.time()
    last_print = 0

    while len(completed_metrics) < len(MODELS):
        time.sleep(5)
        now = time.time()

        # Check inference processes
        for name, (proc, fp, m) in inference_procs.items():
            if name not in completed_inference:
                poll = proc.poll()
                if poll is not None:
                    fp.close()
                    completed_inference.add(name)
                    print(f"\n[INFERENCE FINISHED] {m['display_name']} (Exit code {poll}). Launching parallel metric scoring...")

                    # Launch metrics computation
                    metric_log = LOGS_DIR / f"{name}_metrics.log"
                    m_fp = open(metric_log, "a")
                    m_cmd = [
                        str(METRIC_PYTHON),
                        str(SCRIPT_DIR / "compute_metrics_parallel.py"),
                        "--pred-dir", str(OUT_ROOT / name / "predictions"),
                        "--data-root", str(DATA_ROOT),
                        "--output-dir", str(OUT_ROOT / name),
                        "--model-name", m["display_name"],
                        "--workers", "11"
                    ]
                    m_proc = subprocess.Popen(m_cmd, stdout=m_fp, stderr=subprocess.STDOUT)
                    metric_procs[name] = (m_proc, m_fp, m)

        # Check metric processes
        for name, (proc, fp, m) in list(metric_procs.items()):
            if name not in completed_metrics:
                poll = proc.poll()
                if poll is not None:
                    fp.close()
                    completed_metrics.add(name)
                    print(f"\n[METRICS COMPLETE] {m['display_name']} scored successfully!")

        # Periodic status banner (every 20s)
        if now - last_print >= 20:
            last_print = now
            elapsed = int(now - start_time)
            mins, secs = divmod(elapsed, 60)
            print(f"\n--- Progress Update [{mins}m {secs}s elapsed] ---")
            
            # Print GPU stats
            gpu_stats = get_gpu_stats()
            if gpu_stats:
                gpu_str = " | ".join([f"GPU {idx}: {mem_used}/{mem_total}MB ({util}%)" for idx, mem_used, mem_total, util in gpu_stats[:4]])
                print(f"[GPUs 0-3] {gpu_str}")

            for m in MODELS:
                name = m["name"]
                c_langs, total_s = check_progress(name)
                status_str = "Scoring" if name in completed_inference and name not in completed_metrics else \
                             "DONE" if name in completed_metrics else "Inferring"
                print(f"  * {m['display_name']:<18} [GPU {m['gpu']}]: {c_langs}/11 langs ({total_s}/2200 samples) [{status_str}]")

    print("\n" + "=" * 80)
    print("ALL 11 MODELS INFERENCE & METRIC EVALUATION COMPLETED!")
    print("Consolidating leaderboard...")
    print("=" * 80)

    # Run consolidation
    subprocess.run([
        str(METRIC_PYTHON),
        str(SCRIPT_DIR / "consolidate_leaderboard.py"),
        "--base-dir", str(OUT_ROOT)
    ], check=True)

    # Print final matrix
    subprocess.run([
        str(METRIC_PYTHON),
        str(SCRIPT_DIR / "print_matrix.py"),
        str(OUT_ROOT / "final_leaderboard.json")
    ], check=True)

if __name__ == "__main__":
    main()
