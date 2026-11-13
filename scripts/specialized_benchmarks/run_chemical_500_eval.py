#!/usr/bin/env python3
"""
Master Evaluation Pipeline for Chemical Tables (500 samples).
Runs all 11 TSR models across available NVIDIA A100 GPUs.
Pins all CPU tasks with taskset to cores 0-15 so other users are not affected.
Computes TEDS-Struct and GriTS-Top metrics and updates the leaderboard.
"""

import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path

DATA_ROOT = Path("/home/tanish/scratch/tanishq/table-understanding/data/Table Datasets/pubtabnet/chemical_tables_500")
OUT_ROOT = Path("/scratch/tanish/tanishq/table-understanding/repos/synthetic-table-generation/eval_chemical_tables_500")
SCRIPT_DIR = Path("/home/tanish/scratch/tanishq/table-understanding/repos/inference_scripts/batch_eval")
CONDA_BASE = Path("/home/tanish/scratch/miniconda3/envs")
METRIC_PYTHON = CONDA_BASE / "tables-detr" / "bin" / "python"
TOTAL_SAMPLES = 500

LOGS_DIR = OUT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {"key": "tatr", "name": "TATR v1.1-All"},
    {"key": "slanet", "name": "SLANet-plus"},
    {"key": "unitable", "name": "UniTable"},
    {"key": "tableformer", "name": "TableFormerV2"},
    {"key": "lore", "name": "LORE-TSR"},
    {"key": "cycle_centernet", "name": "Cycle-CenterNet"},
    {"key": "mtltabnet", "name": "MTL-TabNet"},
    {"key": "pixtab", "name": "PIX-TAB"},
    {"key": "tflop", "name": "TFLOP"},
    {"key": "sprint", "name": "SPRINT"},
    {"key": "smoldocling", "name": "SmolDocling-256M"},
]

# Work queues per GPU (GPUs 0, 1, 2, 3 are completely free)
GPU_QUEUES = {
    0: [
        {
            "model_key": "tatr",
            "name": "TATR v1.1-All",
            "python": CONDA_BASE / "tables-detr" / "bin" / "python",
            "script": SCRIPT_DIR / "batch_tatr.py",
        },
        {
            "model_key": "mtltabnet",
            "name": "MTL-TabNet",
            "python": CONDA_BASE / "mtltabnet" / "bin" / "python",
            "script": SCRIPT_DIR / "batch_mtltabnet.py",
        },
    ],
    1: [
        {
            "model_key": "slanet",
            "name": "SLANet-plus",
            "python": CONDA_BASE / "slanet" / "bin" / "python",
            "script": SCRIPT_DIR / "batch_slanet.py",
        },
        {
            "model_key": "lore",
            "name": "LORE-TSR",
            "python": CONDA_BASE / "lore-tsr" / "bin" / "python",
            "script": SCRIPT_DIR / "batch_lore.py",
        },
        {
            "model_key": "cycle_centernet",
            "name": "Cycle-CenterNet",
            "python": CONDA_BASE / "cycle-centernet" / "bin" / "python",
            "script": SCRIPT_DIR / "batch_cycle_centernet.py",
        },
    ],
    2: [
        {
            "model_key": "unitable",
            "name": "UniTable",
            "python": CONDA_BASE / "unitable" / "bin" / "python",
            "script": SCRIPT_DIR / "batch_unitable.py",
        },
        {
            "model_key": "tableformer",
            "name": "TableFormerV2",
            "python": CONDA_BASE / "tableformer" / "bin" / "python",
            "script": SCRIPT_DIR / "batch_tableformer.py",
        },
    ],
    3: [
        {
            "model_key": "tflop",
            "name": "TFLOP",
            "python": CONDA_BASE / "tflop" / "bin" / "python",
            "script": SCRIPT_DIR / "batch_tflop.py",
        },
        {
            "model_key": "sprint",
            "name": "SPRINT",
            "python": CONDA_BASE / "sprint" / "bin" / "python",
            "script": SCRIPT_DIR / "batch_sprint.py",
        },
        {
            "model_key": "pixtab",
            "name": "PIX-TAB",
            "python": CONDA_BASE / "mtltabnet" / "bin" / "python",
            "script": SCRIPT_DIR / "batch_pixtab.py",
        },
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

def check_and_copy_tatr():
    test_tatr = Path("/scratch/tanish/tanishq/table-understanding/repos/synthetic-table-generation/test_tatr_chem/predictions/chemical.jsonl")
    dest_tatr = OUT_ROOT / "tatr" / "predictions" / "chemical.jsonl"
    dest_tatr.parent.mkdir(parents=True, exist_ok=True)
    if not dest_tatr.exists() and test_tatr.exists():
        lines = count_lines(test_tatr)
        if lines >= TOTAL_SAMPLES:
            shutil.copy(test_tatr, dest_tatr)
            print(f"[TATR] Copied completed {lines} predictions from test directory.")

def print_leaderboard():
    leaderboard = []
    for m in MODELS:
        mkey = m["key"]
        summary_file = OUT_ROOT / mkey / "summary.json"
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
                "simple_grits": by_comp.get("simple", {}).get("grits_top", 0.0),
                "complex_grits": by_comp.get("complex", {}).get("grits_top", 0.0),
            }
            leaderboard.append(row)
        except Exception as e:
            print(f"Error reading summary for {mkey}: {e}")

    leaderboard.sort(key=lambda x: x["teds_struct"], reverse=True)
    out_file = OUT_ROOT / "final_leaderboard.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    print("\n" + "=" * 95)
    print(f"{'Rank':<5} | {'Model Name':<18} | {'TEDS-Struct':<12} | {'GriTS-Top':<10} | {'GriTS P':<8} | {'GriTS R':<8} | {'Status'}")
    print("-" * 95)
    for i, r in enumerate(leaderboard, 1):
        print(f"{i:<5} | {r['model_name']:<18} | {r['teds_struct']*100:6.2f}%     | {r['grits_top']*100:6.2f}%   | {r['grits_precision']*100:5.2f}%  | {r['grits_recall']*100:5.2f}%  | Scored ({r['total_records']}/500)")
    print("=" * 95 + "\n")

def main():
    print("=" * 80)
    print("CHEMICAL TABLES (500) TSR EVALUATION PIPELINE")
    print(f"Data Root:   {DATA_ROOT}")
    print(f"Output Root: {OUT_ROOT}")
    print(f"Models:      {len(MODELS)}")
    print("CPU affinity pinned to cores 0-15 via taskset")
    print("=" * 80)

    for m in MODELS:
        (OUT_ROOT / m["key"] / "predictions").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / m["key"] / "scores").mkdir(parents=True, exist_ok=True)

    check_and_copy_tatr()

    # Active running processes per GPU: {gpu_id: (proc, fp, job_dict)}
    active_gpu_procs = {}
    gpu_queue_idx = {gpu: 0 for gpu in GPU_QUEUES}

    # SmolDocling sharded state
    smoldocling_shards_launched = False
    smoldocling_procs = {} # shard_id: (proc, fp, gpu)
    smoldocling_done = False

    # Metric scoring processes: {model_key: (proc, fp)}
    active_metric_procs = {}
    completed_scoring = set()

    start_time = time.time()
    last_status = 0

    while len(completed_scoring) < len(MODELS):
        time.sleep(5)
        now = time.time()
        check_and_copy_tatr()

        # Check GPU queue processes
        for gpu in list(active_gpu_procs.keys()):
            proc, fp, job = active_gpu_procs[gpu]
            poll = proc.poll()
            if poll is not None:
                fp.close()
                mkey = job["model_key"]
                lines = count_lines(OUT_ROOT / mkey / "predictions" / "chemical.jsonl")
                print(f"[JOB COMPLETE] {job['name']} on GPU {gpu} finished (code {poll}, {lines}/500 samples).")
                del active_gpu_procs[gpu]

        # Launch next job in GPU queue if idle
        for gpu, jobs in GPU_QUEUES.items():
            if gpu not in active_gpu_procs and gpu not in [p[2] for p in smoldocling_procs.values()]:
                idx = gpu_queue_idx[gpu]
                while idx < len(jobs):
                    cand_job = jobs[idx]
                    mkey = cand_job["model_key"]
                    pred_file = OUT_ROOT / mkey / "predictions" / "chemical.jsonl"
                    if count_lines(pred_file) >= TOTAL_SAMPLES:
                        idx += 1
                        gpu_queue_idx[gpu] = idx
                        continue
                    break

                if idx < len(jobs):
                    job = jobs[idx]
                    mkey = job["model_key"]
                    log_file = LOGS_DIR / f"{mkey}_inference.log"
                    fp = open(log_file, "a", buffering=1)
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    env["PYTHONUNBUFFERED"] = "1"

                    cmd = [
                        "taskset", "-c", "0-15",
                        str(job["python"]),
                        str(job["script"]),
                        "--data-root", str(DATA_ROOT),
                        "--output-dir", str(OUT_ROOT / mkey),
                        "--device", "cuda:0",
                        "--language", "chemical"
                    ]
                    print(f"[LAUNCH] {job['name']} -> GPU {gpu} (taskset 0-15)")
                    proc = subprocess.Popen(cmd, env=env, stdout=fp, stderr=subprocess.STDOUT)
                    active_gpu_procs[gpu] = (proc, fp, job)
                    gpu_queue_idx[gpu] = idx + 1

        # Check if all initial GPU queue jobs are done, then launch SmolDocling sharded
        all_initial_done = all(gpu_queue_idx[gpu] >= len(GPU_QUEUES[gpu]) and gpu not in active_gpu_procs for gpu in GPU_QUEUES)
        smol_pred = OUT_ROOT / "smoldocling" / "predictions" / "chemical.jsonl"

        if count_lines(smol_pred) >= TOTAL_SAMPLES:
            smoldocling_done = True

        if all_initial_done and not smoldocling_shards_launched and not smoldocling_done:
            print("\n>>> Launching SmolDocling-256M sharded across 4 GPUs (GPUs 0, 1, 2, 3)...")
            smol_py = CONDA_BASE / "smoldocling" / "bin" / "python"
            smol_script = SCRIPT_DIR / "batch_smoldocling.py"
            num_shards = 4
            for s_id in range(num_shards):
                gpu = s_id
                log_file = LOGS_DIR / f"smoldocling_shard_{s_id}.log"
                fp = open(log_file, "a", buffering=1)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["PYTHONUNBUFFERED"] = "1"

                cmd = [
                    "taskset", "-c", "0-15",
                    str(smol_py),
                    str(smol_script),
                    "--data-root", str(DATA_ROOT),
                    "--output-dir", str(OUT_ROOT / "smoldocling"),
                    "--device", "cuda:0",
                    "--language", "chemical",
                    "--shard-id", str(s_id),
                    "--num-shards", str(num_shards)
                ]
                print(f"[LAUNCH] SmolDocling Shard {s_id}/4 -> GPU {gpu} (taskset 0-15)")
                proc = subprocess.Popen(cmd, env=env, stdout=fp, stderr=subprocess.STDOUT)
                smoldocling_procs[s_id] = (proc, fp, gpu)
            smoldocling_shards_launched = True

        # Check SmolDocling shard processes
        if smoldocling_shards_launched and not smoldocling_done:
            all_shards_finished = True
            for s_id, (proc, fp, gpu) in list(smoldocling_procs.items()):
                poll = proc.poll()
                if poll is not None:
                    fp.close()
                else:
                    all_shards_finished = False

            if all_shards_finished:
                print("\n[SMOLDOCLING SHARDS COMPLETE] Merging shard predictions into chemical.jsonl...")
                shard_files = [OUT_ROOT / "smoldocling" / "predictions" / f"chemical_shard_{s}.jsonl" for s in range(4)]
                combined_records = {}
                for sf in shard_files:
                    if sf.exists():
                        with open(sf, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    try:
                                        d = json.loads(line)
                                        combined_records[d["image_id"]] = line.strip()
                                    except Exception:
                                        pass
                with open(smol_pred, "w", encoding="utf-8") as f:
                    for img_id in sorted(combined_records.keys()):
                        f.write(combined_records[img_id] + "\n")
                print(f"[SMOLDOCLING] Merged {len(combined_records)} records into {smol_pred.name}.")
                smoldocling_done = True

        # Check if any model's predictions are complete and ready for metric evaluation
        for m in MODELS:
            mkey = m["key"]
            if mkey not in active_metric_procs and mkey not in completed_scoring:
                pred_file = OUT_ROOT / mkey / "predictions" / "chemical.jsonl"
                lines = count_lines(pred_file)
                if lines >= TOTAL_SAMPLES:
                    print(f"\n>>> [SCORING LAUNCH] {m['name']} completed inference ({lines} samples). Launching metric evaluation...")
                    metric_log = LOGS_DIR / f"{mkey}_metrics.log"
                    m_fp = open(metric_log, "a", buffering=1)
                    m_cmd = [
                        "taskset", "-c", "0-15",
                        str(METRIC_PYTHON),
                        str(SCRIPT_DIR / "compute_metrics_parallel.py"),
                        "--pred-dir", str(OUT_ROOT / mkey / "predictions"),
                        "--data-root", str(DATA_ROOT),
                        "--output-dir", str(OUT_ROOT / mkey),
                        "--model-name", m["name"],
                        "--languages", "chemical",
                        "--workers", "4"
                    ]
                    m_proc = subprocess.Popen(m_cmd, stdout=m_fp, stderr=subprocess.STDOUT)
                    active_metric_procs[mkey] = (m_proc, m_fp)

        # Check metric processes
        for mkey, (proc, fp) in list(active_metric_procs.items()):
            poll = proc.poll()
            if poll is not None:
                fp.close()
                completed_scoring.add(mkey)
                del active_metric_procs[mkey]
                print(f"\n[METRICS COMPLETE] {mkey} scored successfully!")
                print_leaderboard()

        # Periodic status report every 30s
        if now - last_status >= 30:
            last_status = now
            elapsed = int(now - start_time)
            mins, secs = divmod(elapsed, 60)
            print(f"--- Benchmark Progress [{mins:02d}m {secs:02d}s elapsed] [{len(completed_scoring)}/{len(MODELS)} models scored] ---")
            for m in MODELS:
                mkey = m["key"]
                lines = count_lines(OUT_ROOT / mkey / "predictions" / "chemical.jsonl")
                if mkey in completed_scoring:
                    st = "DONE (Scored)"
                elif mkey in active_metric_procs:
                    st = "Scoring..."
                elif lines >= TOTAL_SAMPLES:
                    st = "Inference Done (Queued for Scoring)"
                elif mkey in [job["model_key"] for proc, fp, job in active_gpu_procs.values()]:
                    st = f"Inferring ({lines}/500)"
                elif smoldocling_shards_launched and mkey == "smoldocling":
                    st = "Inferring (4 shards running)"
                else:
                    st = "Queued"
                print(f"  * {m['name']:<18}: {st}")

    print("\n" + "=" * 80)
    print("ALL 11 MODELS COMPLETE AND SCORED ON CHEMICAL TABLES (500)!")
    print("=" * 80)
    print_leaderboard()

if __name__ == "__main__":
    main()
