#!/usr/bin/env python3
"""
Orchestrates parallel SmolDocling-256M inference across GPUs 3, 4, 5, 6 pinned to CPU cores 0-15.
Environment: smoldocling
"""

import sys
import subprocess
import time
from pathlib import Path

DATA_ROOT = "/home/tanish/scratch/tanishq/table-understanding/data/Table Datasets/pubtabnet/multilingual_500_master_run"
OUT_ROOT = "/scratch/tanish/tanishq/table-understanding/repos/synthetic-table-generation/eval_multilingual_500_master_run/smoldocling"
SCRIPT_PATH = "/scratch/tanish/tanishq/table-understanding/repos/synthetic-table-generation/run_smoldocling_fixed.py"
PYTHON_BIN = "/home/tanish/scratch/miniconda3/envs/smoldocling/bin/python"

GPU_ALLOCATIONS = [
    {"gpu": "cuda:3", "cpus": "0-3", "langs": ["asm_Beng", "ben_Beng", "eng_Latn"]},
    {"gpu": "cuda:4", "cpus": "4-7", "langs": ["guj_Gujr", "hin_Deva", "kan_Knda"]},
    {"gpu": "cuda:5", "cpus": "8-11", "langs": ["mal_Mlym", "ory_Orya", "pan_Guru"]},
    {"gpu": "cuda:6", "cpus": "12-15", "langs": ["tam_Taml", "tel_Telu"]},
]

processes = []
log_files = []

for alloc in GPU_ALLOCATIONS:
    gpu = alloc["gpu"]
    cpus = alloc["cpus"]
    langs = alloc["langs"]
    log_file = Path(OUT_ROOT) / f"smoldocling_{gpu.replace(':', '_')}.log"
    log_fp = open(log_file, "w")
    log_files.append(log_fp)
    
    cmd = [
        "taskset", "-c", cpus,
        PYTHON_BIN, SCRIPT_PATH,
        "--data-root", DATA_ROOT,
        "--output-dir", OUT_ROOT,
        "--device", gpu,
        "--batch-size", "8"
    ]
    for l in langs:
        cmd.extend(["--language", l])
        
    print(f"Launching SmolDocling worker on {gpu} (CPUs {cpus}) for languages: {langs}")
    p = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT)
    processes.append((gpu, p))

# Wait for all processes to complete
exit_codes = {}
for gpu, p in processes:
    code = p.wait()
    exit_codes[gpu] = code
    print(f"Worker on {gpu} exited with code {code}")

for fp in log_files:
    fp.close()

if any(code != 0 for code in exit_codes.values()):
    print("Some workers failed!")
    sys.exit(1)
else:
    print("All SmolDocling workers finished successfully!")
