#!/usr/bin/env bash
# Evaluate MoPatch (Motion Patches, CVPR 2024) on the leaderboard HumanML3D test pairs.
#
# Usage:
#   bash leaderboard/methods/mopatch/run.sh /path/to/HumanML3D
#
# /path/to/HumanML3D must contain new_joints/ (every file is used for Mean_raw/Std_raw).
# The Python environment (MotionPatches' requirements.txt) is prepared by the user;
# set PYTHON to use a specific interpreter and DEVICE (e.g. cpu, cuda:0) to choose the device.
set -euo pipefail

HUMANML3D=$(realpath "${1:?usage: run.sh /path/to/HumanML3D}")
PYTHON=${PYTHON:-python}
DEVICE_ARG=${DEVICE:+--device $DEVICE}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEADERBOARD=$(cd "$HERE/../.." && pwd)
WORK=${WORK_DIR:-$LEADERBOARD/.cache/mopatch}
REPO_URL=https://github.com/line/MotionPatches.git
COMMIT=1ce221f5b2b0bd23158ab3372377eae1cdb0cad4
HF_REPO=line-corporation/MotionPatches
HF_REVISION=5001206455aa2d97c931791fc7e5cec3193b9200

mkdir -p "$WORK"
REPO=$WORK/MotionPatches

# 1. Clone the official code at a fixed commit
if [ ! -d "$REPO/.git" ]; then
    git clone "$REPO_URL" "$REPO"
fi
git -C "$REPO" checkout -q "$COMMIT"
cd "$REPO"

# 2. Download the pretrained model from Hugging Face
CKPT=checkpoints/pretrained/HumanML3D/best_model.pt
if [ ! -f "$CKPT" ]; then
    mkdir -p "$(dirname "$CKPT")"
    curl -fL "https://huggingface.co/$HF_REPO/resolve/$HF_REVISION/HumanML3D/best_model.pt" -o "$CKPT.part"
    mv "$CKPT.part" "$CKPT"
fi

# 3. Export the embeddings of the test pairs (Mean_raw/Std_raw are computed once and cached)
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$HERE/export_embeddings.py" \
    --humanml3d "$HUMANML3D" \
    --pairs "$LEADERBOARD/benchmark/humanml3d_test_pairs.json" \
    --stats_dir "$WORK/stats" \
    --out "$WORK/embeddings.npz" $DEVICE_ARG

# 4. Compute the metrics
"$PYTHON" "$LEADERBOARD/eval/evaluate.py" "$WORK/embeddings.npz" "$LEADERBOARD/results/mopatch.json"
