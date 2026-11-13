#!/usr/bin/env bash
# Universal TSR Model Runner
# Usage: ./run_model.sh <model_name> <image_path> [gpu_id]
set -e

MODEL=$1
IMG=$2
GPU_ID=${3:-1}

if [ -z "$MODEL" ] || [ -z "$IMG" ]; then
    echo "Usage: $0 <model_name> <image_path> [gpu_id (default: 1)]"
    echo ""
    echo "Available models:"
    echo "  - unitable        (AR Transformer decoder: HTML + BBoxes)"
    echo "  - sprint          (AR OTSL sequence model)"
    echo "  - mtltabnet       (AR multi-task decoder: HTML + BBoxes)"
    echo "  - pixtab          (AR PAPP-OTSL speculative model)"
    echo "  - tflop           (AR layout/table transformer)"
    echo "  - tatr            (Detection: DETR row/col/cell queries)"
    echo "  - slanet          (AR structure-location alignment network)"
    echo "  - lore            (Spatial CenterNet + Logical regression)"
    echo "  - cycle-centernet (Spatial vertex keypoint pairing)"
    echo "  - tableformer     (OTSL + attention cell bbox head)"
    echo "  - smoldocling     (Compact 256M Document VLM)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="/home/tanish/scratch/miniconda3/envs"

export CUDA_VISIBLE_DEVICES=$GPU_ID

case "$MODEL" in
    unitable)
        $CONDA_BASE/unitable/bin/python "$SCRIPT_DIR/run_unitable.py" --image "$IMG"
        ;;
    sprint)
        $CONDA_BASE/sprint/bin/python "$SCRIPT_DIR/run_sprint.py" --image "$IMG"
        ;;
    mtltabnet)
        $CONDA_BASE/mtltabnet/bin/python "$SCRIPT_DIR/run_mtltabnet.py" --image "$IMG"
        ;;
    pixtab)
        $CONDA_BASE/mtltabnet/bin/python "$SCRIPT_DIR/run_pixtab.py" --image "$IMG"
        ;;
    tflop)
        $CONDA_BASE/tflop/bin/python "$SCRIPT_DIR/run_tflop.py" --image "$IMG"
        ;;
    tatr)
        $CONDA_BASE/tables-detr/bin/python "$SCRIPT_DIR/run_tatr.py" --image "$IMG"
        ;;
    slanet)
        $CONDA_BASE/slanet/bin/python "$SCRIPT_DIR/run_slanet.py" --image "$IMG"
        ;;
    lore)
        $CONDA_BASE/lore-tsr/bin/python "$SCRIPT_DIR/run_lore.py" --image "$IMG"
        ;;
    cycle-centernet)
        $CONDA_BASE/cycle-centernet/bin/python "$SCRIPT_DIR/run_cycle_centernet.py" --image "$IMG"
        ;;
    tableformer)
        $CONDA_BASE/tableformer/bin/python "$SCRIPT_DIR/run_tableformer.py" --image "$IMG"
        ;;
    smoldocling)
        $CONDA_BASE/smoldocling/bin/python "$SCRIPT_DIR/run_smoldocling.py" --image "$IMG"
        ;;
    *)
        echo "Error: Unknown model '$MODEL'."
        exit 1
        ;;
esac
