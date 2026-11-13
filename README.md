# Table Structure Recognition (TSR): Consolidated Inference, Model & Benchmark Suite

Comprehensive, unified evaluation suite containing all **11 Table Structure Recognition (TSR) models**, their **single-table and batch inference scripts**, **checkpoint weights and configs**, **synthetic data generators**, **bug fixes and workers**, and **evaluation predictions and leaderboards** across multilingual, chemical, blank cell, gibberish stress-test, and degraded benchmark datasets.

---

## 1. Directory Structure

```
tsr_eval_and_inference_suite/
├── README.md                                  # Complete documentation guide
├── models/
│   ├── checkpoints/                           # Model weights & trained checkpoints (3.0 GB)
│   │   ├── TATR-v1.1-All-msft.pth             # Table Transformer weights (111 MB)
│   │   ├── unitable_large_structure.pt        # UniTable Large Structure weights (474 MB)
│   │   ├── mtltabnet_epoch_19.pth             # MTL-TabNet epoch 19 checkpoint (1.1 GB)
│   │   ├── pixtab_epoch_25.pth                # PIX-TAB epoch 25 checkpoint (1.0 GB)
│   │   ├── SPRINT_merged.pt                   # SPRINT weights (305 MB)
│   │   └── upstage-TFLOP/                     # Upstage TFLOP weights & tokenizer (422 MB)
│   ├── configs/                               # Model configuration files
│   │   ├── structure_config.json              # TATR model architecture config
│   │   ├── table_master_*.py                  # MMOCR / Table-MASTER pipeline configs
│   │   └── tflop_config.json                  # TFLOP mBART decoder config
│   └── model_registry.json                    # Unified metadata registry across all 11 models
│
├── scripts/
│   ├── single_inference/                      # Per-model single-image inference runners
│   │   ├── run_tatr.py                        # TATR inference runner
│   │   ├── run_unitable.py                    # UniTable inference runner
│   │   ├── run_unitable_full.py               # UniTable full pipeline runner
│   │   ├── run_mtltabnet.py                   # MTL-TabNet PAPP-OTSL inference runner
│   │   ├── run_pixtab.py                      # PIX-TAB inference runner
│   │   ├── run_slanet.py                      # SLANet-plus inference runner
│   │   ├── run_lore.py                        # LORE-TSR inference runner
│   │   ├── run_sprint.py                      # SPRINT inference runner
│   │   ├── run_tableformer.py                 # TableFormerV2 inference runner
│   │   ├── run_cycle_centernet.py             # Cycle-CenterNet inference runner
│   │   ├── run_smoldocling.py                 # SmolDocling-256M inference runner
│   │   ├── run_tflop.py                       # TFLOP inference runner
│   │   └── run_model.sh                       # Unified shell wrapper for running any model
│   │
│   ├── batch_evaluation/                      # Full multi-table / multi-language batch runners
│   │   ├── batch_tatr.py, batch_unitable.py, batch_mtltabnet.py, batch_pixtab.py...
│   │   ├── run_full_evaluation.py             # Orchestrates batch runs across models
│   │   ├── run_degraded_evaluation.py         # Orchestrates runs on degraded datasets
│   │   ├── compute_metrics.py                 # TEDS-Struct and GriTS metric scorer
│   │   ├── compute_metrics_parallel.py        # Multi-worker parallel metric scorer
│   │   ├── consolidate_leaderboard.py         # Aggregates scores into JSON leaderboards
│   │   └── print_matrix.py                    # Formats markdown leaderboards & matrices
│   │
│   ├── specialized_benchmarks/                # Specialized domain & stress-test benchmark runners
│   │   ├── run_chemical_benchmark.py          # Chemical table benchmark runner
│   │   ├── run_chemical_500_eval.py           # Chemical 500 images benchmark evaluator
│   │   ├── run_dual_chemical_benchmarks.py    # Chemical SMILES & Combined dual runner
│   │   ├── run_blank_cells_benchmark.py       # Blank cells stress-test benchmark runner
│   │   ├── run_gibberish_benchmarks.py        # Gibberish text stress-test benchmark runner
│   │   ├── fast_score_chemical.py             # Fast parallel scorer for chemical tables
│   │   └── score_chemical_flexible.py         # Flexible HTML-aligned scorer for chemical tables
│   │
│   ├── data_generation/                       # Synthetic dataset generation scripts
│   │   ├── generate_chemical_tables.py        # Generates chemical tables with rendered molecular images
│   │   ├── generate_chemical_tables_smiles.py # Generates chemical tables with raw SMILES text strings
│   │   ├── generate_chemical_tables_combined.py # Generates chemical tables combining images & SMILES
│   │   ├── generate_blank_cells_tables.py     # Generates tables with 30%–60% random blank cells
│   │   └── generate_gibberish_tables.py       # Generates tables with random or all gibberish cells
│   │
│   ├── fixes_and_workers/                     # Diagnosed fixes & multi-GPU bbox workers
│   │   ├── run_smoldocling_fixed.py           # Official DocTags OTSL parser (no text slicing)
│   │   ├── run_sprint_grid_aligned.py         # docTR normalization + TATR grid alignment
│   │   ├── launch_unitable_bbox_all_gpus.py   # 6-GPU parallel UniTable cell bbox runner
│   │   ├── run_unitable_bbox_worker.py        # UniTable cell bbox extraction worker
│   │   └── compute_additional_metrics.py      # Computes TEDS-100 and COCO Cell-IoU AP
│   │
│   └── utils/                                 # Shared helper utilities
│       ├── otsl_utils.py                      # OTSL sequence normalization & HTML conversion
│       ├── draw_bboxes.py                     # Table cell & structure visualizer
│       └── create_dashboard.py                # Generates interactive HTML comparison dashboards
│
└── results/                                   # Benchmark predictions, logs & leaderboards
    ├── multilingual_500_master_run/           # 5,500 table master benchmark (11 langs x 500) (468 MB)
    │   ├── final_leaderboard.json             # Overall & per-language benchmark metrics
    │   ├── final_leaderboard_extended.json    # Extended leaderboard with TEDS-100 & Cell-AP
    │   ├── logs/                              # Complete execution stdout/stderr logs
    │   ├── unitable_bbox_shards/              # Sharded bounding box predictions
    │   └── <model_name>/                      # Per-model predictions & scores per language
    │
    ├── chemical_tables_500_images/            # Chemical experiment 1: Rendered molecular images (83 MB)
    ├── chemical_tables_500_smiles/            # Chemical experiment 2: SMILES formula strings (39 MB)
    ├── chemical_tables_500_combined/          # Chemical experiment 3: Mixed Images + SMILES (51 MB)
    ├── blank_cells_tables_500/                # Blank cells experiment: 30%–60% empty cells (32 MB)
    ├── gibberish_random_cells_500/            # Gibberish experiment 1: Random 30%–60% gibberish cells (39 MB)
    ├── gibberish_all_cells_500/               # Gibberish experiment 2: 100% gibberish cells (39 MB)
    ├── hardest_200_multilingual_synthetic/    # 2,200 table challenging benchmark (277 MB)
    ├── multilingual_200_degraded/             # 2,200 degraded table benchmark (115 MB)
    └── dashboards/                            # Visual dashboards
        └── comparison_dashboard.html          # Interactive side-by-side visualization
```

---

## 2. Benchmark Results & Leaderboards

### A. Gibberish Text Stress-Test Experiments (500 Tables Each)

Evaluates whether models genuinely detect visual cell grid boundaries or rely on NLP/language semantics.
*Results from [`results/gibberish_random_cells_500/`](file:///home/tanish/tsr_eval_and_inference_suite/results/gibberish_random_cells_500/final_leaderboard.json) and [`results/gibberish_all_cells_500/`](file:///home/tanish/tsr_eval_and_inference_suite/results/gibberish_all_cells_500/final_leaderboard.json)*:

| Rank | Model | Random Cells (30%-60% Gibberish) | All Cells (100% Gibberish) | GriTS-Top (All Gibberish) | Semantic Dependency Analysis |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | **UniTable** | **95.73%** | **96.00%** | **97.29%** | **Completely Invariant**: Operates on pure visual patch embeddings. Scores actually increase slightly without linguistic noise. |
| 2 | **TATR v1.1-All** | **92.05%** | **92.05%** | **94.07%** | **Zero Degradation**: Pure object detector queries table geometry, oblivious to text semantics. |
| 3 | **SLANet-plus** | **87.06%** | **86.75%** | **94.77%** | Near-perfect grid topology retention. |
| 4 | **LORE-TSR** | **86.35%** | **85.71%** | **93.00%** | Spatial and logic points cluster robustly on arbitrary character strings. |
| 5 | **TableFormerV2** | **85.53%** | **85.17%** | **91.89%** | Resilient coordinate attention. |
| 6 | **MTL-TabNet** | **89.68%** | **85.08%** | **88.43%** | Small dip (~4.6%) when all cells lack real word tokens. |
| 7 | **PIX-TAB** | **83.08%** | **78.89%** | **85.72%** | Moderate dip (~4.2%) due to character width variance in random strings. |
| 8 | **SPRINT** | **74.20%** | **72.06%** | **78.11%** | Small desynchronization on variable-length random token sequences. |
| 9 | **Cycle-CenterNet** | **39.67%** | **37.69%** | **37.83%** | Unaffected by text (limited by wired line detection). |
| 10 | **TFLOP** | **32.50%** | **32.13%** | **33.05%** | Layout pointer bypass limitation remains. |

---

### B. Chemical Table Experiments (All 3 Runs)

Chemical tables represent a major domain challenge: cells can contain either graphical molecular depictions, textual chemical notations (SMILES), or a mixture of both.

#### 1. Chemical Tables: Rendered Molecular Structure Images (500 Tables)
*Leaderboard file: [`results/chemical_tables_500_images/final_leaderboard.json`](file:///home/tanish/tsr_eval_and_inference_suite/results/chemical_tables_500_images/final_leaderboard.json)*

| Rank | Model | TEDS-Struct | GriTS-Top | Key Observation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **TATR v1.1-All** | **80.16%** | **85.08%** | **Top Performer**: Object detection bounding box heads are robust against embedded non-text imagery. |
| 2 | **UniTable** | **70.73%** | **77.39%** | Vision encoder struggles slightly with pure visual molecules replacing text tokens. |
| 3 | **LORE-TSR** | **62.66%** | **74.54%** | Point cascade retains good grid topology despite visual content. |
| 4 | **SLANet-plus** | **62.13%** | **74.07%** | Fast structure attention handles image cell boundaries well. |
| 5 | **TableFormerV2** | **55.45%** | **64.37%** | Degrades when text tokens are absent. |
| 6 | **MTL-TabNet** | **48.46%** | **55.89%** | Positional prompts shift due to large graphical cell contents. |
| 7 | **SPRINT** | **45.92%** | **35.31%** | Sequence length diverges on image cells. |
| 8 | **PIX-TAB** | **41.31%** | **51.74%** | PAPP coordinates distorted by molecular bounding boxes. |
| 9 | **Cycle-CenterNet** | **35.58%** | **34.65%** | Fails on wireless table borders. |
| 10 | **TFLOP** | **17.54%** | **24.32%** | Bypassed pointer mechanism drops further on graphical content. |
| 11 | **SmolDocling-256M** | **12.17%** | **10.90%** | Uncorrected token slicing corrupted on visual cells. |

---

#### 2. Chemical Tables: SMILES Formula Strings (500 Tables)
*Leaderboard file: [`results/chemical_tables_500_smiles/final_leaderboard.json`](file:///home/tanish/tsr_eval_and_inference_suite/results/chemical_tables_500_smiles/final_leaderboard.json)*

| Rank | Model | TEDS-Struct | GriTS-Top | Key Observation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **UniTable** | **95.46%** | **96.57%** | **Dominant Performer**: Textual SMILES strings behave identically to standard text tokens. |
| 2 | **TATR v1.1-All** | **89.64%** | **91.54%** | Consistent spatial structure extraction. |
| 3 | **TableFormerV2** | **85.15%** | **92.09%** | Strong recovery on text-only tabular data. |
| 4 | **SLANet-plus** | **83.95%** | **92.81%** | High topology GriTS score. |
| 5 | **MTL-TabNet** | **83.81%** | **87.15%** | Standard PAPP-OTSL performance restored. |
| 6 | **LORE-TSR** | **82.73%** | **90.65%** | Solid spatial point recovery. |
| 7 | **SPRINT** | **76.37%** | **83.25%** | Restored grid length alignment. |
| 8 | **PIX-TAB** | **75.38%** | **81.94%** | High structural accuracy with text SMILES. |
| 9 | **Cycle-CenterNet** | **39.71%** | **39.01%** | Architecture limited to wired borders. |
| 10 | **TFLOP** | **31.61%** | **32.94%** | Dummy bbox input limitation remains. |

---

#### 3. Chemical Tables: Combined Molecular Images & SMILES Strings (500 Tables)
*Leaderboard file: [`results/chemical_tables_500_combined/final_leaderboard.json`](file:///home/tanish/tsr_eval_and_inference_suite/results/chemical_tables_500_combined/final_leaderboard.json)*

| Rank | Model | TEDS-Struct | GriTS-Top | Key Observation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **TATR v1.1-All** | **82.62%** | **87.10%** | **Leader**: Robust against multimodal mixture of images and text. |
| 2 | **UniTable** | **78.68%** | **84.24%** | Strong performance across the combined multimodal cells. |
| 3 | **LORE-TSR** | **70.83%** | **82.01%** | Dual logic-point network adapts well to mixed modalities. |
| 4 | **SLANet-plus** | **68.76%** | **80.36%** | High topological F1 score. |
| 5 | **MTL-TabNet** | **67.10%** | **72.27%** | Decent recovery on mixed modality cells. |
| 6 | **SPRINT** | **62.74%** | **60.24%** | Intermediate performance between pure text and pure image. |
| 7 | **PIX-TAB** | **60.71%** | **71.12%** | Moderate degradation on image-containing rows. |
| 8 | **TableFormerV2** | **59.33%** | **68.15%** | Sensitive to non-textual cell regions. |
| 9 | **Cycle-CenterNet** | **38.81%** | **39.87%** | Baseline performance on wireless tables. |
| 10 | **TFLOP** | **22.99%** | **29.38%** | Low score due to missing upstream detection coordinates. |

---

### C. Blank Cells Table Experiment (500 Tables)

Evaluates model resilience when **30% to 60% of table cells are completely empty/blank**.
*Leaderboard file: [`results/blank_cells_tables_500/final_leaderboard.json`](file:///home/tanish/tsr_eval_and_inference_suite/results/blank_cells_tables_500/final_leaderboard.json)*

| Rank | Model | TEDS-Struct | GriTS-Top | Resilience to Empty Cells |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **UniTable** | **86.75%** | **90.78%** | **Most Resilient**: Robust autoregressive empty-cell token handling (`<td></td>`). |
| 2 | **MTL-TabNet** | **79.06%** | **84.53%** | Positional prompts anchor empty cells even without text. |
| 3 | **LORE-TSR** | **77.05%** | **85.53%** | High GriTS topology; spatial point clustering identifies cell presence. |
| 4 | **SLANet-plus** | **76.61%** | **86.76%** | Excellent grid topology preservation. |
| 5 | **PIX-TAB** | **71.62%** | **81.14%** | Coordinate anchors prevent column collapses. |
| 6 | **TATR v1.1-All** | **70.21%** | **76.48%** | Struggles slightly when empty cell regions lack feature activations. |
| 7 | **TableFormerV2** | **63.40%** | **73.21%** | Attention tends to merge adjacent blank cells. |
| 8 | **SPRINT** | **50.85%** | **56.97%** | Token count desynchronizes significantly when many cells are blank. |
| 9 | **Cycle-CenterNet** | **37.15%** | **40.39%** | Baseline wireless table limitation. |
| 10 | **TFLOP** | **32.80%** | **39.35%** | Bypassed pointer mechanism drops further. |

---

### D. Multilingual Master Benchmark (`multilingual_500_master_run`)
*Leaderboard file: [`results/multilingual_500_master_run/final_leaderboard_extended.json`](file:///home/tanish/tsr_eval_and_inference_suite/results/multilingual_500_master_run/final_leaderboard_extended.json)*

5,500 tables across 11 languages (Assamese, Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Odia, Punjabi, Tamil, Telugu):

| Rank | Model | TEDS-Struct | TEDS-100 (Perfect) | GriTS-Top | GriTS-100 | Cell AP50 | Cell mAP |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **UniTable** | **95.01%** | **60.09%** | **96.36%** | **65.15%** | 86.4% | 78.1% |
| 2 | **TATR v1.1-All** | **88.89%** | **34.95%** | **91.19%** | **42.00%** | 84.7% | 76.9% |
| 3 | **MTL-TabNet** | **88.29%** | **44.93%** | **90.78%** | **48.05%** | 81.2% | 73.5% |
| 4 | **LORE-TSR** | **85.72%** | 0.00% | **92.42%** | **42.38%** | 79.5% | 71.0% |
| 5 | **SLANet-plus** | **85.02%** | 0.00% | **93.39%** | **51.96%** | 80.1% | 72.4% |
| 6 | **SPRINT** | **81.49%** | 0.00% | **89.15%** | **36.31%** | N/A | N/A |
| 7 | **TableFormerV2** | **80.41%** | 0.00% | **87.93%** | **28.58%** | 75.3% | 67.8% |
| 8 | **PIX-TAB** | **78.25%** | 0.00% | **86.09%** | **37.89%** | 73.8% | 65.2% |
| 9 | **Cycle-CenterNet** | **35.95%** | 0.00% | **33.59%** | 1.47% | 22.4% | 18.1% |
| 10 | **TFLOP** | **30.81%** | 0.00% | **33.59%** | 0.00% | N/A | N/A |
| 11 | **SmolDocling-256M** | **26.98%** | 0.00% | **31.25%** | 4.85% | N/A | N/A |

---

## 3. Synthetic Data Generation Guide

Scripts located in [`scripts/data_generation/`](file:///home/tanish/tsr_eval_and_inference_suite/scripts/data_generation/):

```bash
# 1. Generate Gibberish Tables (Random or All Cells):
conda activate translate
python scripts/data_generation/generate_gibberish_tables.py \
    --num-tables 500 \
    --mode all \
    --seed 2026 \
    --output-dir "./output/gibberish_all_tables"

# 2. Generate Chemical Tables with Rendered Molecular Images:
python scripts/data_generation/generate_chemical_tables.py \
    --num-tables 500 \
    --workers 8 \
    --fixed-mol-prob 0.40 \
    --degrade \
    --output-dir "./output/chemical_images_tables"

# 3. Generate Chemical Tables with SMILES String Formulas:
python scripts/data_generation/generate_chemical_tables_smiles.py \
    --num-tables 500 \
    --workers 8 \
    --seed 2026 \
    --max-smiles-len 45 \
    --output-dir "./output/chemical_smiles_tables"

# 4. Generate Chemical Tables Combining Images and SMILES:
python scripts/data_generation/generate_chemical_tables_combined.py \
    --num-tables 500 \
    --workers 8 \
    --seed 2026 \
    --max-smiles-len 45 \
    --img-ratio 0.50 \
    --output-dir "./output/chemical_combined_tables"

# 5. Generate Tables with 30%-60% Random Blank Cells:
python scripts/data_generation/generate_blank_cells_tables.py \
    --num-tables 500 \
    --seed 2026 \
    --output-dir "./output/blank_cells_tables"
```

---

## 4. Specialized Benchmark Runners

Scripts located in [`scripts/specialized_benchmarks/`](file:///home/tanish/tsr_eval_and_inference_suite/scripts/specialized_benchmarks/):

```bash
# Run all models on the gibberish benchmarks:
python scripts/specialized_benchmarks/run_gibberish_benchmarks.py \
    --dataset-random "/path/to/gibberish_random_tables_500" \
    --dataset-all "/path/to/gibberish_all_tables_500" \
    --output-root "./eval_output/gibberish"

# Run all models on the dual chemical benchmarks (SMILES & Combined):
python scripts/specialized_benchmarks/run_dual_chemical_benchmarks.py \
    --gpus 0 1 2 3 \
    --output-root "./eval_output/chemical"

# Run all models on the blank cells benchmark:
python scripts/specialized_benchmarks/run_blank_cells_benchmark.py \
    --dataset-dir "/path/to/blank_cells_tables_500" \
    --output-dir "./eval_output/blank_cells"
```
