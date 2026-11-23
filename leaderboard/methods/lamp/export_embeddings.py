"""Export LaMP (ICLR 2025) text/motion embeddings for the leaderboard test pairs.

Run from the root of a clone of https://github.com/gentlefress/LaMP with the HumanML3D
checkpoint (h3d-qformer.tar) and bert-base-uncased in ./bert/ (run.sh does this):

    python export_embeddings.py --humanml3d /path/to/HumanML3D --ckpt h3d-qformer.tar \
        --mean mean.npy --std std.npy --pairs humanml3d_test_pairs.json --out embeddings.npz

The released evaluation code cannot be used as is: models/t2m_eval_wrapper.py imports the
missing package models.q_former (the code lives in models/lamp), and eval_retrieval.py uses
the Guo et al. evaluator instead of LaMP. This script builds the same modules as
EvaluatorModelWrapper_w_LaMP with the classmethods of models/lamp/QFormer_Base.py and loads
every module from the checkpoint with strict key matching.

Embeddings follow the contrastive loss of training (models/lamp/motiongpm_lamp.py, LaMP.forward):
  - Text: pre_caption (lowercase, punctuation removed) of the benchmark caption, BERT tokens
    padded/truncated to 32, the Q-Former text CLS feature projected by text_proj.
  - Motion: new_joint_vecs/<id>.npy normalized with --mean/--std. The checkpoints do not store
    the statistics used in training. train_vq.py reads the dataset Mean.npy/Std.npy, but every
    released evaluation (utils/eval_t2m.py) feeds motions normalized with the evaluator meta
    (Guo et al. Comp_v6) to the model, and only the evaluator meta reproduces the paper (small
    batches, T2M R@1 66.2 vs 67.18 in the paper; 42.7 with the dataset Mean.npy/Std.npy).
    run.sh therefore passes the meta of the Guo et al. checkpoint Comp_v6_KLD01.
    The length is floored to a multiple of 4, clips longer than 196 frames use their first
    196 frames, and shorter ones are zero-padded to 196 frames as in data/t2m_dataset.py.
    The output is one vector per query token, [49, 512].
  - Similarity: "max_cosine", the maximum cosine over the query tokens, as in training.
"""
import argparse
import json
import re
from os.path import join as pjoin
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from models.lamp.QFormer_Base import QFormer_Base

UNIT_LENGTH = 4
MAX_MOTION_LENGTH = 196
MAX_TXT_LEN = 32
NUM_QUERY_TOKEN = 49  # train_lamp.py builds the model with num_query_token=49
# Motion encoder (VQ-VAE encoder) settings of EvaluatorModelWrapper_w_LaMP
ENCODER_OPT = SimpleNamespace(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                              dilation_growth_rate=3, vq_norm=None)


def pre_caption(caption):
    """Copied from Text2MotionDataset.pre_caption in data/t2m_dataset.py."""
    caption = re.sub(r"([.!\"()*#:;~])", " ", caption.lower())
    caption = re.sub(r"\s{2,}", " ", caption)
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")
    caption_words = caption.split(" ")
    if len(caption_words) > 32:
        caption = " ".join(caption_words[:32])
    return caption


def sub_state_dict(state_dict, prefix):
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humanml3d", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mean", required=True)
    ap.add_argument("--std", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    device = torch.device(args.device)

    # ---- model: the modules of EvaluatorModelWrapper_w_LaMP ----
    qformer, _ = QFormer_Base.init_Qformer(num_query_token=NUM_QUERY_TOKEN, vision_width=1408,
                                           cross_attention_freq=2)
    tokenizer = QFormer_Base.init_tokenizer()
    qformer.resize_token_embeddings(len(tokenizer))
    motion_encoder = QFormer_Base.init_motion_encoder("motion_encoder", ENCODER_OPT, "t2m")
    text_proj, motion_proj = nn.Linear(768, 512), nn.Linear(768, 512)

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["motion_qformer"]
    qformer.load_state_dict(sub_state_dict(sd, "Qformer."), strict=True)
    motion_encoder.load_state_dict(sub_state_dict(sd, "motion_encoder."), strict=True)
    text_proj.load_state_dict(sub_state_dict(sd, "text_proj."), strict=True)
    motion_proj.load_state_dict(sub_state_dict(sd, "motion_proj."), strict=True)
    query_tokens = sd["query_tokens"].to(device)
    motion_projection = sd["motion_projection"].to(device)
    assert query_tokens.shape == (1, NUM_QUERY_TOKEN, 768), query_tokens.shape
    for m in (qformer, motion_encoder, text_proj, motion_proj):
        m.to(device).eval()

    mean, std = np.load(args.mean), np.load(args.std)

    def motion_input(keyid):
        motion = np.load(pjoin(args.humanml3d, "new_joint_vecs", keyid + ".npy"))
        m_length = (min(len(motion), MAX_MOTION_LENGTH) // UNIT_LENGTH) * UNIT_LENGTH
        assert m_length >= UNIT_LENGTH, f"motion too short: {keyid}"
        motion = (motion[:m_length] - mean) / std
        padded = np.zeros((MAX_MOTION_LENGTH, motion.shape[1]), dtype=np.float32)
        padded[:m_length] = motion
        return torch.from_numpy(padded)

    pairs = json.load(open(args.pairs, encoding="utf-8"))["pairs"]
    ids = [p["id"] for p in pairs]
    text_emb, motion_emb = [], []
    with torch.no_grad():
        for s in tqdm(range(0, len(pairs), args.batch_size), desc="encode"):
            chunk = pairs[s:s + args.batch_size]

            tokens = tokenizer([pre_caption(p["text"]) for p in chunk], padding="max_length",
                               truncation=True, max_length=MAX_TXT_LEN,
                               return_tensors="pt").to(device)
            text_out = qformer.bert(tokens.input_ids, attention_mask=tokens.attention_mask,
                                    return_dict=True)
            text_emb.append(F.normalize(text_proj(text_out.last_hidden_state[:, 0, :]), dim=-1).cpu())

            motions = torch.stack([motion_input(p["id"]) for p in chunk]).to(device)
            movements = motion_encoder(motions.permute(0, 2, 1)).permute(0, 2, 1)
            motion_embeds = movements @ motion_projection
            motion_atts = torch.ones(motion_embeds.size()[:-1], dtype=torch.long, device=device)
            query_out = qformer.bert(query_embeds=query_tokens.expand(len(chunk), -1, -1),
                                     encoder_hidden_states=motion_embeds,
                                     encoder_attention_mask=motion_atts,
                                     use_cache=True, return_dict=True)
            motion_emb.append(F.normalize(motion_proj(query_out.last_hidden_state), dim=-1).cpu())

    np.savez(args.out, ids=np.array(ids), text=torch.cat(text_emb).numpy(),
             motion=torch.cat(motion_emb).numpy(), similarity=np.array("max_cosine"))
    print(f"saved {len(ids)} pairs to {args.out}")


if __name__ == "__main__":
    main()
