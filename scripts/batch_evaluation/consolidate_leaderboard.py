#!/usr/bin/env python3
"""
Leaderboard aggregator for TSR Multilingual Evaluation (PubTabNet 200 subset).
Reads summary.json from each model directory under benchmark_eval_200.
"""
import argparse
import json
from pathlib import Path

DEFAULT_BASE_DIR = Path("/home/tanish/scratch/tanishq/table-understanding/repos/benchmark_eval_200")
ALL_LANGUAGES = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]

def main():
    parser = argparse.ArgumentParser(description="Leaderboard aggregator")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Base directory containing model subdirs")
    args = parser.parse_args()
    base_dir = args.base_dir

    models = sorted([d.name for d in base_dir.iterdir() if d.is_dir() and d.name != "logs"])
    leaderboard = []

    for model in models:
        summary_file = base_dir / model / "summary.json"
        if not summary_file.exists():
            continue
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            ov = data.get("overall", {})
            langs = data.get("languages", {})

            row = {
                "model_key": model,
                "model_name": data.get("model", model),
                "total_records": data.get("total_records", 0),
                "valid": ov.get("valid", 0),
                "teds_struct": ov.get("teds_struct", 0.0),
                "grits_top": ov.get("grits_top", 0.0),
                "grits_precision": ov.get("grits_top_precision", 0.0),
                "grits_recall": ov.get("grits_top_recall", 0.0),
                "per_language": {}
            }

            for l in ALL_LANGUAGES:
                l_data = langs.get(l, {})
                row["per_language"][l] = {
                    "teds_struct": l_data.get("teds_struct", 0.0),
                    "grits_top": l_data.get("grits_top", 0.0),
                    "grits_precision": l_data.get("grits_top_precision", 0.0),
                    "grits_recall": l_data.get("grits_top_recall", 0.0),
                }

            leaderboard.append(row)
        except Exception as e:
            print(f"Error reading {summary_file}: {e}")

    # Sort descending by TEDS-Struct
    leaderboard.sort(key=lambda x: x["teds_struct"], reverse=True)

    out_file = base_dir / "final_leaderboard.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    print(f"Loaded {len(leaderboard)} models into leaderboard:\n")
    print(f"{'Rank':<5} | {'Model Name':<18} | {'TEDS-Struct':<12} | {'GriTS-Top':<10} | {'GriTS P':<8} | {'GriTS R':<8}")
    print("-" * 75)
    for i, r in enumerate(leaderboard, 1):
        print(f"{i:<5} | {r['model_name']:<18} | {r['teds_struct']*100:6.2f}%     | {r['grits_top']*100:6.2f}%   | {r['grits_precision']*100:5.2f}%  | {r['grits_recall']*100:5.2f}%")

if __name__ == "__main__":
    main()
