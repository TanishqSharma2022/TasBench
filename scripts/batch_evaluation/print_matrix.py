#!/usr/bin/env python3
import sys
import json
from pathlib import Path

json_path = sys.argv[1] if len(sys.argv) > 1 else "/home/tanish/scratch/tanishq/table-understanding/repos/benchmark_eval_200/final_leaderboard.json"

with open(json_path) as f:
    leaderboard = json.load(f)

langs = [
    "asm_Beng", "ben_Beng", "eng_Latn", "guj_Gujr", "hin_Deva",
    "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"
]
short_langs = [l.split("_")[0] for l in langs]

print(f"{'Model':<16} | " + " | ".join([f"{l:<6}" for l in short_langs]) + " | Overall")
print("=" * 125)
for m in leaderboard:
    name = m["model_name"]
    teds_cells = [f"{m['per_language'][l]['teds_struct']*100:5.1f}%" for l in langs]
    overall_teds = f"{m['teds_struct']*100:5.1f}%"
    print(f"{name:<16} | " + " | ".join(teds_cells) + f" | {overall_teds} (TEDS)")
    grits_cells = [f"{m['per_language'][l]['grits_top']*100:5.1f}%" for l in langs]
    overall_grits = f"{m['grits_top']*100:5.1f}%"
    print(f"{'  (GriTS-Top)':<16} | " + " | ".join(grits_cells) + f" | {overall_grits}")
    print("-" * 125)
