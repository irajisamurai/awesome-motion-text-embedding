#!/usr/bin/env bash
# Evaluate LaMP (ICLR 2025) on the leaderboard HumanML3D test pairs.
#
# Usage:
#   bash leaderboard/methods/lamp/run.sh /path/to/HumanML3D
#
# /path/to/HumanML3D must contain new_joint_vecs/.
# The Python environment (torch, transformers, numpy, gdown) is prepared by the user;
# set PYTHON to use a specific interpreter and DEVICE (e.g. cpu, cuda:0) to choose the device.
set -euo pipefail

HUMANML3D=$(realpath "${1:?usage: run.sh /path/to/HumanML3D}")
PYTHON=${PYTHON:-python}
DEVICE_ARG=${DEVICE:+--device $DEVICE}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEADERBOARD=$(cd "$HERE/../.." && pwd)
WORK=${WORK_DIR:-$LEADERBOARD/.cache/lamp}
REPO_URL=https://github.com/gentlefress/LaMP.git
COMMIT=48deb9ebb76085ffe91d432e3a3085962aa37d0f
CKPT_URL=https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/lamp/h3d-qformer.tar
BERT_REPO=google-bert/bert-base-uncased
BERT_REVISION=86b5e0934494bd15c9632b12f734a8a67f723594
GUO_GDRIVE_ID=1IgrFCnxeg4olBtURUHimzS03ZI0df_6W   # Guo et al. HumanML3D checkpoints (also used by methods/guo)

mkdir -p "$WORK"
REPO=$WORK/LaMP

# 1. Clone the official code at a fixed commit
if [ ! -d "$REPO/.git" ]; then
    git clone "$REPO_URL" "$REPO"
fi
git -C "$REPO" checkout -q "$COMMIT"
cd "$REPO"

# 2. Download the HumanML3D checkpoint (a torch.save file named .tar; the server often drops
#    the connection, so the download is resumed until it completes)
CKPT=$WORK/h3d-qformer.tar
if [ ! -f "$CKPT" ]; then
    for i in 1 2 3 4 5 6 7 8 9 10; do
        curl -fL -C - --retry 3 "$CKPT_URL" -o "$CKPT.part" && break
        echo "download interrupted (attempt $i), resuming" >&2
    done
    "$PYTHON" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).testzip()" "$CKPT.part"
    mv "$CKPT.part" "$CKPT"
fi

# 3. bert-base-uncased in ./bert (QFormer_Base loads the tokenizer and config from "bert")
mkdir -p bert
for f in config.json vocab.txt tokenizer_config.json pytorch_model.bin; do
    if [ ! -f "bert/$f" ]; then
        curl -fL "https://huggingface.co/$BERT_REPO/resolve/$BERT_REVISION/$f" -o "bert/$f.part"
        mv "bert/$f.part" "bert/$f"
    fi
done

# 4. Normalization statistics: the evaluator meta of Guo et al. (see export_embeddings.py)
META=$WORK/evaluator_meta
if [ ! -f "$META/mean.npy" ] || [ ! -f "$META/std.npy" ]; then
    ZIP=$LEADERBOARD/.cache/guo/t2m_checkpoints.zip   # reuse the download of methods/guo if any
    if [ ! -f "$ZIP" ]; then
        ZIP=$WORK/t2m_checkpoints.zip
        if [ ! -f "$ZIP" ]; then
            if ! "$PYTHON" -c "import gdown, sys; assert gdown.download(sys.argv[1], sys.argv[2], quiet=False)" \
                    "https://drive.google.com/uc?id=$GUO_GDRIVE_ID" "$ZIP.part"; then
                echo "Download failed. Old gdown versions cannot fetch from Google Drive" \
                     "(4.4.0 fails, 6.4.1 works): run 'pip install -U gdown', or download" \
                     "https://drive.google.com/file/d/$GUO_GDRIVE_ID to $ZIP" >&2
                exit 1
            fi
            mv "$ZIP.part" "$ZIP"
        fi
    fi
    mkdir -p "$META"
    unzip -q -o -j "$ZIP" t2m/Comp_v6_KLD01/meta/mean.npy t2m/Comp_v6_KLD01/meta/std.npy -d "$META"
fi

# 5. Export the embeddings of the test pairs
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$HERE/export_embeddings.py" \
    --humanml3d "$HUMANML3D" \
    --ckpt "$CKPT" \
    --mean "$META/mean.npy" --std "$META/std.npy" \
    --pairs "$LEADERBOARD/benchmark/humanml3d_test_pairs.json" \
    --out "$WORK/embeddings.npz" $DEVICE_ARG

# 6. Compute the metrics
"$PYTHON" "$LEADERBOARD/eval/evaluate.py" "$WORK/embeddings.npz" "$LEADERBOARD/results/lamp.json"
