#!/usr/bin/env bash
# Evaluate TMR (ICCV 2023) on the leaderboard HumanML3D test pairs.
#
# Usage:
#   bash leaderboard/methods/tmr/run.sh /path/to/HumanML3D /path/to/pose_data
#
# /path/to/HumanML3D must contain texts/. /path/to/pose_data is the pose_data folder made by
# raw_pose_processing.ipynb of the HumanML3D repo (TMR computes its features from it).
# The Python environment (TMR's requirements.txt and gdown) is prepared by the user;
# set PYTHON to use a specific interpreter and DEVICE (e.g. cpu, cuda:0) to choose the device.
set -euo pipefail

HUMANML3D=$(realpath "${1:?usage: run.sh /path/to/HumanML3D /path/to/pose_data}")
POSE_DATA=$(realpath "${2:?usage: run.sh /path/to/HumanML3D /path/to/pose_data}")
PYTHON=${PYTHON:-python}
DEVICE_ARG=${DEVICE:+--device $DEVICE}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEADERBOARD=$(cd "$HERE/../.." && pwd)
WORK=${WORK_DIR:-$LEADERBOARD/.cache/tmr}
REPO_URL=https://github.com/Mathux/TMR.git
COMMIT=6d74688730d15d43b0a755ce2b0e1f2d76138fc1
MODELS_GDRIVE_ID=1n6kRb-d2gKsk8EXfFULFIpaUKYcnaYmm   # from prepare/download_pretrain_models.sh
MODELS_MD5=7b6d8814f9c1ca972f62852ebb6c7a6f

mkdir -p "$WORK"
REPO=$WORK/TMR

# 1. Clone the official code at a fixed commit
if [ ! -d "$REPO/.git" ]; then
    git clone "$REPO_URL" "$REPO"
fi
git -C "$REPO" checkout -q "$COMMIT"
cd "$REPO"

# 2. Download the pretrained models
if [ ! -f models/tmr_humanml3d_guoh3dfeats/config.json ]; then
    TGZ=$WORK/tmr_models.tgz
    if [ ! -f "$TGZ" ]; then
        if ! "$PYTHON" -c "import gdown, sys; assert gdown.download(sys.argv[1], sys.argv[2], quiet=False)" \
                "https://drive.google.com/uc?id=$MODELS_GDRIVE_ID" "$TGZ.part"; then
            echo "Download failed. Old gdown versions cannot fetch from Google Drive" \
                 "(4.4.0 fails, 6.4.1 works): run 'pip install -U gdown', or download" \
                 "https://drive.google.com/file/d/$MODELS_GDRIVE_ID to $TGZ" >&2
            exit 1
        fi
        mv "$TGZ.part" "$TGZ"
    fi
    echo "$MODELS_MD5  $TGZ" | md5sum -c -
    tar xzf "$TGZ" -C "$REPO"
fi
[ -f models/tmr_humanml3d_guoh3dfeats/config.json ] || { echo "models/tmr_humanml3d_guoh3dfeats not found after extraction" >&2; exit 1; }

# 3. Compute TMR's Guo features from pose_data (files that already exist are skipped)
mkdir -p datasets/motions
ln -sfn "$POSE_DATA" datasets/motions/pose_data
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m prepare.compute_guoh3dfeats

# 4. Export the embeddings of the test pairs
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$HERE/export_embeddings.py" \
    --humanml3d "$HUMANML3D" \
    --pairs "$LEADERBOARD/benchmark/humanml3d_test_pairs.json" \
    --out "$WORK/embeddings.npz" $DEVICE_ARG

# 5. Compute the metrics
"$PYTHON" "$LEADERBOARD/eval/evaluate.py" "$WORK/embeddings.npz" "$LEADERBOARD/results/tmr.json"
