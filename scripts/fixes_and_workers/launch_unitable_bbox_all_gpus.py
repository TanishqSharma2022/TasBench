#!/usr/bin/env python3
"""
Orchestrate UniTable bbox inference across idle GPUs (0, 1, 2, 3, 6, 7).
Once complete, updates unitable/predictions/*.jsonl and computes Cell-IOU AP for UniTable.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

DATA_ROOT = Path("/scratch/tanish/tanishq/table-understanding/data/Table Datasets/pubtabnet/multilingual_500_master_run")
EVAL_ROOT = Path("/scratch/tanish/tanishq/table-understanding/repos/synthetic-table-generation/eval_multilingual_500_master_run")
WORKER_SCRIPT = Path("/scratch/tanish/tanishq/table-understanding/repos/synthetic-table-generation/run_unitable_bbox_worker.py")
PYTHON = Path("/scratch/tanish/miniconda3/envs/unitable/bin/python")

GPUS = [0, 1, 2, 3, 6, 7]
ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

TMP_DIR = EVAL_ROOT / "unitable_bbox_shards"
TMP_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"Collecting all tasks from {DATA_ROOT}...")
    all_tasks = []
    for lang in ALL_LANGUAGES:
        ann_file = DATA_ROOT / "annotations" / f"{lang}.jsonl"
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    all_tasks.append((lang, item["image_id"], item["image"]))

    print(f"Total tasks: {len(all_tasks)} across {len(ALL_LANGUAGES)} languages.")
    
    # Check already processed
    done_ids = set()
    for f in TMP_DIR.glob("boxes_*.jsonl"):
        with open(f, "r", encoding="utf-8") as fp:
            for l in fp:
                if l.strip():
                    done_ids.add(json.loads(l)["image_id"])

    remaining = [t for t in all_tasks if t[1] not in done_ids]
    print(f"Remaining tasks to process: {len(remaining)} (Already done: {len(done_ids)})")

    if not remaining:
        print("All tasks already completed!")
        merge_results()
        return

    # Shard remaining tasks across GPUs
    num_gpus = len(GPUS)
    shards = [[] for _ in range(num_gpus)]
    for idx, task in enumerate(remaining):
        shards[idx % num_gpus].append(task)

    procs = []
    for i, gpu in enumerate(GPUS):
        task_list = shards[i]
        if not task_list:
            continue
        task_file = TMP_DIR / f"tasks_gpu_{gpu}.json"
        with open(task_file, "w", encoding="utf-8") as fp:
            json.dump(task_list, fp)
        log_file = TMP_DIR / f"log_gpu_{gpu}.log"
        print(f"Launching GPU {gpu} with {len(task_list)} tasks -> {log_file.name}")
        fp_log = open(log_file, "w")
        p = subprocess.Popen([
            str(PYTHON), str(WORKER_SCRIPT),
            "--gpu", str(gpu),
            "--task-file", str(task_file),
            "--out-dir", str(TMP_DIR)
        ], stdout=fp_log, stderr=subprocess.STDOUT)
        procs.append((p, fp_log, gpu))

    # Wait for completion
    print("Waiting for UniTable bbox inference across all GPUs...")
    t0 = time.time()
    for p, fp_log, gpu in procs:
        p.wait()
        fp_log.close()
        print(f"GPU {gpu} finished with code {p.returncode}.")

    print(f"Inference complete in {time.time() - t0:.2f}s!")
    merge_results()

def merge_results():
    print("Merging bbox shards into unitable/predictions/*.jsonl...")
    box_map = {}
    for f in TMP_DIR.glob("boxes_*.jsonl"):
        with open(f, "r", encoding="utf-8") as fp:
            for l in fp:
                if l.strip():
                    rec = json.loads(l)
                    box_map[rec["image_id"]] = rec["cell_bboxes"]

    print(f"Total bbox records mapped: {len(box_map)}")

    pred_dir = EVAL_ROOT / "unitable" / "predictions"
    for lang in ALL_LANGUAGES:
        f_pred = pred_dir / f"{lang}.jsonl"
        if not f_pred.exists():
            continue
        with open(f_pred, "r", encoding="utf-8") as fp:
            lines = [json.loads(l) for l in fp if l.strip()]

        updated_count = 0
        for item in lines:
            img_id = item["image_id"]
            if img_id in box_map:
                item["cell_bboxes"] = box_map[img_id]
                updated_count += 1

        with open(f_pred, "w", encoding="utf-8") as fp:
            for item in lines:
                fp.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  * Updated {lang}: {updated_count}/{len(lines)} records with bboxes.")

    print("UniTable prediction files updated with bounding boxes successfully!")

if __name__ == "__main__":
    main()
