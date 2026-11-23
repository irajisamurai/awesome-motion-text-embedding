#!/usr/bin/env bash
# Evaluate SGAR (NeurIPS 2025) on the leaderboard HumanML3D test pairs (global embeddings only).
#
# Usage:
#   bash leaderboard/methods/sgar/run.sh /path/to/HumanML3D
#
# /path/to/HumanML3D must contain new_joints/ (every file is used for Mean_raw/Std_raw).
# The Python environment is prepared by the user. SGAR's requirements.txt pins timm==0.9.5,
# but the code calls VisionTransformer.forward_intermediates, which needs timm >= 1.0.
# Set PYTHON to use a specific interpreter and DEVICE (e.g. cpu, cuda:0) to choose the device.
set -euo pipefail

HUMANML3D=$(realpath "${1:?usage: run.sh /path/to/HumanML3D}")
PYTHON=${PYTHON:-python}
DEVICE_ARG=${DEVICE:+--device $DEVICE}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEADERBOARD=$(cd "$HERE/../.." && pwd)
WORK=${WORK_DIR:-$LEADERBOARD/.cache/sgar}
REPO_URL=https://github.com/JHang2020/SGAR.git
COMMIT=72006c31e97e0de23b63dc35accf0b8e10637397
HF_REPO=Jiahang-HF/SGAR
HF_REVISION=15890373bfe85a149c8565bce204629b01e484a7

mkdir -p "$WORK"
REPO=$WORK/SGAR

# 1. Clone the official code at a fixed commit
if [ ! -d "$REPO/.git" ]; then
    git clone "$REPO_URL" "$REPO"
fi
git -C "$REPO" checkout -q "$COMMIT"
cd "$REPO"

# 2. Download the pretrained model from Hugging Face (the zip also has the part texts, unused here)
CKPT=checkpoints/pretrained/HumanML3D/best_model.pt
if [ ! -f "$CKPT" ]; then
    ZIP=$WORK/SGAR_pretrained.zip
    if [ ! -f "$ZIP" ]; then
        curl -fL "https://huggingface.co/$HF_REPO/resolve/$HF_REVISION/SGAR_pretrained.zip" -o "$ZIP.part"
        mv "$ZIP.part" "$ZIP"
    fi
    mkdir -p "$(dirname "$CKPT")"
    unzip -q -o -j "$ZIP" SGAR_pretrained/HumanML3D/best_model.pt -d "$(dirname "$CKPT")"
fi

# 3. Export the embeddings of the test pairs (Mean_raw/Std_raw are computed once and cached)
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$HERE/export_embeddings.py" \
    --humanml3d "$HUMANML3D" \
    --pairs "$LEADERBOARD/benchmark/humanml3d_test_pairs.json" \
    --stats_dir "$WORK/stats" \
    --out "$WORK/embeddings.npz" $DEVICE_ARG

# 4. Compute the metrics
"$PYTHON" "$LEADERBOARD/eval/evaluate.py" "$WORK/embeddings.npz" "$LEADERBOARD/results/sgar.json"
