"""Export Guo et al. (CVPR 2022) text/motion embeddings for the leaderboard test pairs.

Run from the root of a clone of https://github.com/EricGuo5513/text-to-motion
with the HumanML3D checkpoints in ./checkpoints/t2m/ (run.sh does this):

    python export_embeddings.py --humanml3d /path/to/HumanML3D \
        --pairs humanml3d_test_pairs.json --out embeddings.npz

Inputs follow the original evaluator (networks/evaluator_wrapper.py, data/dataset.py):
  - Text: the word/POS tokens of the first line of texts/<id>.txt (the same line as the
    benchmark caption), wrapped with sos/eos and padded or cropped to 20 tokens, with GloVe vectors.
  - Motion: new_joint_vecs/<id>.npy normalized with checkpoints/t2m/Comp_v6_KLD01/meta mean/std.
    The length is floored to a multiple of 4, and clips longer than 196 frames use their
    first 196 frames (the original crops at a random position).
  - Similarity: negative Euclidean distance (the original ranks by Euclidean distance).
"""
import argparse
import json
import os
from os.path import join as pjoin
from types import SimpleNamespace

import numpy as np
import torch

from networks.evaluator_wrapper import build_models
from utils.word_vectorizer import POS_enumerator, WordVectorizer

UNIT_LENGTH = 4
MAX_MOTION_LENGTH = 196
MAX_TEXT_LEN = 20


def text_inputs(tokens, w_vectorizer):
    if len(tokens) < MAX_TEXT_LEN:
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens = tokens + ["unk/OTHER"] * (MAX_TEXT_LEN + 2 - sent_len)
    else:
        tokens = ["sos/OTHER"] + tokens[:MAX_TEXT_LEN] + ["eos/OTHER"]
        sent_len = len(tokens)
    word_embs, pos_ohots = zip(*(w_vectorizer[t] for t in tokens))
    return np.stack(word_embs), np.stack(pos_ohots), sent_len


def motion_inputs(motion, mean, std):
    m_length = min(len(motion), MAX_MOTION_LENGTH)
    m_length = (m_length // UNIT_LENGTH) * UNIT_LENGTH
    assert m_length >= UNIT_LENGTH, f"motion too short: {len(motion)} frames"
    motion = (motion[:m_length] - mean) / std
    padded = np.zeros((MAX_MOTION_LENGTH, motion.shape[1]), dtype=np.float32)
    padded[:m_length] = motion
    return padded, m_length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humanml3d", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device(args.device)
    opt = SimpleNamespace(
        dim_pose=263, dim_movement_enc_hidden=512, dim_movement_latent=512,
        dim_word=300, dim_pos_ohot=len(POS_enumerator), dim_text_hidden=512,
        dim_motion_hidden=1024, dim_coemb_hidden=512,
        checkpoints_dir="./checkpoints", dataset_name="t2m", device=device,
    )
    text_enc, motion_enc, movement_enc = build_models(opt)
    for m in (text_enc, motion_enc, movement_enc):
        m.to(device).eval()

    meta = pjoin("checkpoints", "t2m", "Comp_v6_KLD01", "meta")
    mean, std = np.load(pjoin(meta, "mean.npy")), np.load(pjoin(meta, "std.npy"))
    w_vectorizer = WordVectorizer("./glove", "our_vab")

    pairs = json.load(open(args.pairs, encoding="utf-8"))["pairs"]
    ids, texts, motions = [], [], []
    for p in pairs:
        first = open(pjoin(args.humanml3d, "texts", p["id"] + ".txt"), encoding="utf-8") \
            .read().strip().split("\n")[0].split("#")
        assert first[0].strip() == p["text"], f"caption mismatch for {p['id']}"
        texts.append(text_inputs(first[1].split(" "), w_vectorizer))
        motion = np.load(pjoin(args.humanml3d, "new_joint_vecs", p["id"] + ".npy"))
        motions.append(motion_inputs(motion, mean, std))
        ids.append(p["id"])

    def run(order, encode):
        out = [None] * len(order)
        for s in range(0, len(order), args.batch_size):
            chunk = order[s:s + args.batch_size]
            for i, e in zip(chunk, encode(chunk).cpu().numpy()):
                out[i] = e
        return np.stack(out)

    def encode_text(chunk):
        w = torch.from_numpy(np.stack([texts[i][0] for i in chunk])).float().to(device)
        p = torch.from_numpy(np.stack([texts[i][1] for i in chunk])).float().to(device)
        lens = torch.tensor([texts[i][2] for i in chunk])
        return text_enc(w, p, lens)

    def encode_motion(chunk):
        x = torch.from_numpy(np.stack([motions[i][0] for i in chunk])).float().to(device)
        lens = torch.tensor([motions[i][1] for i in chunk]) // UNIT_LENGTH
        return motion_enc(movement_enc(x[..., :-4]), lens)

    # pack_padded_sequence needs each batch sorted by length (descending)
    with torch.no_grad():
        text_emb = run(sorted(range(len(ids)), key=lambda i: -texts[i][2]), encode_text)
        motion_emb = run(sorted(range(len(ids)), key=lambda i: -motions[i][1]), encode_motion)

    np.savez(args.out, ids=np.array(ids), text=text_emb, motion=motion_emb,
             similarity=np.array("neg_euclidean"))
    print(f"saved {len(ids)} pairs to {args.out}")


if __name__ == "__main__":
    main()
