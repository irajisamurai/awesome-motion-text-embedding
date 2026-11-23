#!/usr/bin/env bash
# Evaluate Guo et al. (CVPR 2022) on the leaderboard HumanML3D test pairs.
#
# Usage:
#   bash leaderboard/methods/guo/run.sh /path/to/HumanML3D
#
# /path/to/HumanML3D must contain texts/ and new_joint_vecs/.
# The Python environment (torch, numpy, gdown) is prepared by the user;
# set PYTHON to use a specific interpreter and DEVICE (e.g. cpu, cuda:0) to choose the device.
set -euo pipefail

HUMANML3D=$(realpath "${1:?usage: run.sh /path/to/HumanML3D}")
PYTHON=${PYTHON:-python}
DEVICE_ARG=${DEVICE:+--device $DEVICE}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEADERBOARD=$(cd "$HERE/../.." && pwd)
WORK=${WORK_DIR:-$LEADERBOARD/.cache/guo}
REPO_URL=https://github.com/EricGuo5513/text-to-motion.git
COMMIT=72df96ec453edea2fbe9603b1d58a955eaf71636
CKPT_GDRIVE_ID=1IgrFCnxeg4olBtURUHimzS03ZI0df_6W   # HumanML3D checkpoints from the README

mkdir -p "$WORK"
REPO=$WORK/text-to-motion

# 1. Clone the official code at a fixed commit
if [ ! -d "$REPO/.git" ]; then
    git clone "$REPO_URL" "$REPO"
fi
git -C "$REPO" checkout -q "$COMMIT"

# 2. Download the pretrained models
CKPT=$REPO/checkpoints/t2m/text_mot_match/model/finest.tar
if [ ! -f "$CKPT" ]; then
    # The repo tracks checkpoints as a symlink to /checkpoints/ (for Docker); replace it with a folder
    if [ -L "$REPO/checkpoints" ]; then rm "$REPO/checkpoints"; fi
    mkdir -p "$REPO/checkpoints"
    ZIP=$WORK/t2m_checkpoints.zip
    if [ ! -f "$ZIP" ]; then
        if ! "$PYTHON" -c "import gdown, sys; assert gdown.download(sys.argv[1], sys.argv[2], quiet=False)" \
                "https://drive.google.com/uc?id=$CKPT_GDRIVE_ID" "$ZIP.part"; then
            echo "Download failed. Old gdown versions cannot fetch from Google Drive" \
                 "(4.4.0 fails, 6.4.1 works): run 'pip install -U gdown', or download" \
                 "https://drive.google.com/file/d/$CKPT_GDRIVE_ID to $ZIP" >&2
            exit 1
        fi
        mv "$ZIP.part" "$ZIP"
    fi
    unzip -q -o "$WORK/t2m_checkpoints.zip" -d "$REPO/checkpoints"
fi
for f in "$CKPT" "$REPO/checkpoints/t2m/Comp_v6_KLD01/meta/mean.npy" "$REPO/checkpoints/t2m/Comp_v6_KLD01/meta/std.npy"; do
    [ -f "$f" ] || { echo "missing $f (check the layout of the downloaded zip)" >&2; exit 1; }
done

# 3. Export the embeddings of the test pairs
cd "$REPO"
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$HERE/export_embeddings.py" \
    --humanml3d "$HUMANML3D" \
    --pairs "$LEADERBOARD/benchmark/humanml3d_test_pairs.json" \
    --out "$WORK/embeddings.npz" $DEVICE_ARG

# 4. Compute the metrics
"$PYTHON" "$LEADERBOARD/eval/evaluate.py" "$WORK/embeddings.npz" "$LEADERBOARD/results/guo.json"
