"""Export SGAR (NeurIPS 2025) text/motion embeddings for the leaderboard test pairs.

Run from the root of a clone of https://github.com/JHang2020/SGAR with the pretrained
model in ./checkpoints/pretrained/HumanML3D/ (run.sh does this):

    python export_embeddings.py --humanml3d /path/to/HumanML3D \
        --pairs humanml3d_test_pairs.json --out embeddings.npz --stats_dir DIR

Only the global embeddings are used: the LLM-generated body-part texts and the part-level
similarities of scripts/test.py (eval.part_enhanced) are not used.

Inputs follow the original code (scripts/test.py, dataset/dataset.py):
  - Text: the benchmark caption, tokenized with the DistilBERT tokenizer (padding only,
    as in eval_part of scripts/test.py).
  - Motion: new_joints/<id>.npy (first 22 joints), normalized with Mean_raw/Std_raw, which are
    computed over every file in new_joints/ as in scripts/cal_mean_var.py. The five kinematic
    chains are resampled to 16 points each (use_kinematic), and the clip is cut to its first
    224 frames or zero-padded to 224 frames.
  - Motion embedding: the global token of the motion encoder (encode_motion(...)[0]).
  - Similarity: cosine.
"""
import argparse
import json
import os
from os.path import join as pjoin
from types import SimpleNamespace

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset.dataset import TextMotionPartDataset
from models.clip import ClipModel

CKPT = "checkpoints/pretrained/HumanML3D/best_model.pt"
MAX_MOTION_LENGTH = 224
PART_WEIGHT = 0.3  # train.part_weight of conf/test_config.yaml; > 0 builds the part tokens/heads
# HumanML3D kinematic chains (right arm, left arm, torso, right leg, left leg),
# copied from TextMotionPartDataset.__init__
KINEMATIC_CHAIN = [
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
    [0, 3, 6, 9, 12, 15],
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
]


def mean_std(new_joints_dir, stats_dir):
    """Mean_raw / Std_raw of scripts/cal_mean_var.py (HumanML3D only), cached in stats_dir."""
    mean_path, std_path = pjoin(stats_dir, "Mean_raw.npy"), pjoin(stats_dir, "Std_raw.npy")
    if not (os.path.exists(mean_path) and os.path.exists(std_path)):
        data_list = []
        for file in tqdm(sorted(os.listdir(new_joints_dir)), desc="Mean_raw/Std_raw"):
            data = np.load(pjoin(new_joints_dir, file))
            if np.isnan(data).any() or len(data.shape) != 3:
                continue
            data_list.append(data)
        data = np.concatenate(data_list, axis=0)
        os.makedirs(stats_dir, exist_ok=True)
        np.save(mean_path, data.mean(axis=0))
        np.save(std_path, data.std(axis=0))
    return np.load(mean_path), np.load(std_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humanml3d", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    new_joints_dir = pjoin(args.humanml3d, "new_joints")
    mean, std = mean_std(new_joints_dir, args.stats_dir)

    # The ImageNet weights are not needed: every parameter is overwritten by the checkpoint
    model = ClipModel(motion_encoder_alias="vit_base_patch16_224_in21k",
                      text_encoder_alias="distilbert-base-uncased",
                      motion_encoder_pretrained=False,
                      motion_embedding_dims=768, text_embedding_dims=768,
                      projection_dims=256, patch_size=16, part_contrast=PART_WEIGHT)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    kin = SimpleNamespace(patch_size=16, kinematic_chain=KINEMATIC_CHAIN)

    def motion_input(keyid):
        motion = np.load(pjoin(new_joints_dir, keyid + ".npy"))[:, :22]
        motion = (motion - mean[np.newaxis, ...]) / std[np.newaxis, ...]
        motion = TextMotionPartDataset.use_kinematic(kin, motion)
        if len(motion) >= MAX_MOTION_LENGTH:
            motion = motion[:MAX_MOTION_LENGTH]
        else:
            pad = np.zeros((MAX_MOTION_LENGTH - len(motion),) + motion.shape[1:], dtype=np.float32)
            motion = np.concatenate((motion, pad), axis=0)
        return rearrange(torch.tensor(motion).float(), "t j c -> c t j")

    pairs = json.load(open(args.pairs, encoding="utf-8"))["pairs"]
    ids = [p["id"] for p in pairs]
    text_emb, motion_emb = [], []
    with torch.no_grad():
        for s in tqdm(range(0, len(pairs), args.batch_size), desc="encode"):
            chunk = pairs[s:s + args.batch_size]
            motions = torch.stack([motion_input(p["id"]) for p in chunk]).to(args.device)
            tokens = tokenizer([p["text"] for p in chunk], padding=True,
                               return_tensors="pt").to(args.device)
            motion_emb.append(model.encode_motion(motions)[0].cpu())
            text_emb.append(model.encode_text(tokens).cpu())

    np.savez(args.out, ids=np.array(ids), text=torch.cat(text_emb).numpy(),
             motion=torch.cat(motion_emb).numpy(), similarity=np.array("cosine"))
    print(f"saved {len(ids)} pairs to {args.out}")


if __name__ == "__main__":
    main()
