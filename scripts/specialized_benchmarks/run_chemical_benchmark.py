#!/usr/bin/env python3
"""Run all 11 TSR models on the 10 synthetic chemical tables and compute TEDS-Struct and GriTS-Top.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

BASE_DIR = Path("/home/tanish/scratch/tanishq/table-understanding/repos")
SYNTH_DIR = BASE_DIR / "synthetic-table-generation"
INFER_DIR = BASE_DIR / "inference_scripts"
SANITY_DIR = SYNTH_DIR / "sanity_check_chemical_tables"
IMAGES_DIR = SANITY_DIR / "images"
ANN_FILE = SANITY_DIR / "annotations.jsonl"
OUT_BASE = SANITY_DIR / "model_eval_outputs"

GPU_ID = 5
DEV_ARG = "cuda:0"

MODELS = [
    {
        "name": "UniTable",
        "env": "unitable",
        "script": "run_unitable.py",
        "use_dev_arg": True,
        "json_name": "unitable_raw.json",
        "html_name": "unitable.html"
    },
    {
        "name": "SPRINT",
        "env": "sprint",
        "script": "run_sprint.py",
        "use_dev_arg": True,
        "json_name": "sprint_raw.json",
        "html_name": "sprint.html"
    },
    {
        "name": "MTL-TabNet",
        "env": "mtltabnet",
        "script": "run_mtltabnet.py",
        "use_dev_arg": True,
        "json_name": "mtltabnet_raw.json",
        "html_name": "mtltabnet.html"
    },
    {
        "name": "PixTab",
        "env": "mtltabnet",
        "script": "run_pixtab.py",
        "use_dev_arg": True,
        "json_name": "pixtab_raw.json",
        "html_name": "pixtab.html"
    },
    {
        "name": "TFLOP",
        "env": "tflop",
        "script": "run_tflop.py",
        "use_dev_arg": True,
        "json_name": "tflop_raw.json",
        "html_name": "tflop.html"
    },
    {
        "name": "TATR-v1.1",
        "env": "tables-detr",
        "script": "run_tatr.py",
        "use_dev_arg": True,
        "json_name": "tatr_raw.json",
        "html_name": "tatr.html"
    },
    {
        "name": "SLANet-plus",
        "env": "slanet",
        "script": "run_slanet.py",
        "use_dev_arg": False,
        "json_name": "slanet_raw.json",
        "html_name": "slanet.html"
    },
    {
        "name": "LORE-TSR",
        "env": "lore-tsr",
        "script": "run_lore.py",
        "use_dev_arg": False,
        "json_name": "lore_raw.json",
        "html_name": "lore.html"
    },
    {
        "name": "Cycle-CenterNet",
        "env": "cycle-centernet",
        "script": "run_cycle_centernet.py",
        "use_dev_arg": False,
        "json_name": "cycle_centernet_raw.json",
        "html_name": "cycle_centernet.html"
    },
    {
        "name": "TableFormerV2",
        "env": "tableformer",
        "script": "run_tableformer.py",
        "use_dev_arg": True,
        "json_name": "tableformer_raw.json",
        "html_name": "tableformer.html"
    },
    {
        "name": "SmolDocling",
        "env": "smoldocling",
        "script": "run_smoldocling.py",
        "use_dev_arg": True,
        "json_name": "smoldocling_raw.json",
        "html_name": "smoldocling.html"
    }
]

def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    images = sorted(list(IMAGES_DIR.glob("chem_table_*.png")))
    print(f"Found {len(images)} test images in {IMAGES_DIR}")

    predictions_by_model = {}

    for m in MODELS:
        mname = m["name"]
        print(f"\n==========================================")
        print(f"Running inference: {mname} (Env: {m['env']})")
        print(f"==========================================")
        
        py_bin = f"/home/tanish/scratch/miniconda3/envs/{m['env']}/bin/python"
        script_path = str(INFER_DIR / m["script"])
        
        model_out_dir = OUT_BASE / mname
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        preds = {}
        t0 = time.time()
        
        for img in images:
            img_id = img.stem
            out_img_dir = model_out_dir / img_id
            out_img_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if output already exists
            expected_html = out_img_dir / m["html_name"]
            expected_json = out_img_dir / m["json_name"]
            
            if not expected_html.exists():
                cmd = [py_bin, script_path, "--image", str(img), "--output_dir", str(out_img_dir)]
                if m["use_dev_arg"]:
                    cmd.extend(["--device", DEV_ARG])
                
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
                
                try:
                    subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
                except Exception as e:
                    print(f"  [{mname}] Error on {img_id}: {e}")
            
            # Read output
            pred_html = ""
            if expected_html.exists():
                with open(expected_html, "r", encoding="utf-8") as f:
                    pred_html = f.read()
            elif expected_json.exists():
                with open(expected_json, "r", encoding="utf-8") as f:
                    pred_html = json.load(f).get("html", "")
            
            preds[img_id] = pred_html
            print(f"  [{mname}] {img_id}: HTML len {len(pred_html)}")

        predictions_by_model[mname] = preds
        print(f"Completed {mname} in {time.time() - t0:.2f}s")

    # Save all raw predictions
    all_preds_path = OUT_BASE / "all_model_predictions.json"
    with open(all_preds_path, "w", encoding="utf-8") as f:
        json.dump(predictions_by_model, f, indent=2)
    print(f"\nSaved all predictions to {all_preds_path}")

    # Now run evaluation script in tables-detr
    print("\n==========================================")
    print("Computing TEDS-Struct & GriTS-Top metrics...")
    print("==========================================")

    eval_script = SYNTH_DIR / "compute_scores.py"
    tables_detr_py = "/home/tanish/scratch/miniconda3/envs/tables-detr/bin/python"
    
    cmd_eval = [tables_detr_py, str(eval_script)]
    res = subprocess.run(cmd_eval, check=True, text=True)

if __name__ == "__main__":
    main()
