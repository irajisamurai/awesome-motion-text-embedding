"""Export TMR (ICCV 2023) text/motion embeddings for the leaderboard test pairs.

Run from the root of a clone of https://github.com/Mathux/TMR with the pretrained models
in ./models/ and the Guo features in ./datasets/motions/guoh3dfeats/ (run.sh does this):

    python export_embeddings.py --humanml3d /path/to/HumanML3D \
        --pairs humanml3d_test_pairs.json --out embeddings.npz

Inputs follow the original code (retrieval.py, src/data/text_motion.py):
  - Text: the benchmark caption, encoded with TMR's DistilBERT token embeddings.
  - Motion: TMR's Guo features of the whole AMASS sequence (prepare/compute_guoh3dfeats.py),
    cropped to the whole clip of each HumanML3D ID with TMR's AMASSMotionLoader and normalizer.
    The clip range is the TMR annotation that corresponds to a caption without start/end tags.
  - Similarity: cosine (TMR uses the L2-normalized latent means).
"""
import argparse
import json
import math
import os
from os.path import join as pjoin

import numpy as np
import torch

RUN_DIR = "models/tmr_humanml3d_guoh3dfeats"


def whole_clip_range(humanml3d, keyid, tmr_annotations):
    """Start/end (seconds in the AMASS sequence) of the whole HumanML3D clip of keyid."""
    lines = open(pjoin(humanml3d, "texts", keyid + ".txt"), encoding="utf-8").read().strip().split("\n")
    annots = tmr_annotations[keyid]["annotations"]
    assert len(lines) == len(annots), f"caption count mismatch for {keyid}"
    for line, annot in zip(lines, annots):
        start, end = (float(x) for x in line.split("#")[2:4])
        start = 0.0 if math.isnan(start) else start
        end = 0.0 if math.isnan(end) else end
        if start == 0.0 and end == 0.0:
            return annot["start"], annot["end"]
    raise ValueError(f"no caption covers the whole clip of {keyid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humanml3d", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    import src.prepare  # noqa: F401  (registers TMR's config resolvers)
    from hydra.utils import instantiate
    from src.config import read_config
    from src.data.collate import collate_x_dict
    from src.data.motion import AMASSMotionLoader
    from src.data.text import TokenEmbeddings
    from src.load import load_model_from_cfg

    cfg = read_config(RUN_DIR)
    model = load_model_from_cfg(cfg, "last", eval_mode=True, device=args.device)
    normalizer = instantiate(cfg.data.motion_loader.normalizer)
    motion_loader = AMASSMotionLoader(base_dir="datasets/motions/guoh3dfeats", fps=20.0,
                                      normalizer=normalizer)
    text_to_token_emb = TokenEmbeddings(modelname="distilbert-base-uncased",
                                        device=args.device, preload=False)
    tmr_annotations = json.load(open("datasets/annotations/humanml3d/annotations.json"))

    pairs = json.load(open(args.pairs, encoding="utf-8"))["pairs"]
    ids = [p["id"] for p in pairs]

    def encode(x_dicts):
        latents = []
        for s in range(0, len(x_dicts), args.batch_size):
            batch = collate_x_dict(x_dicts[s:s + args.batch_size], device=args.device)
            latents.append(model.encode(batch, sample_mean=True).cpu())
        return torch.cat(latents).numpy()

    with torch.inference_mode():
        motions = []
        for keyid in ids:
            start, end = whole_clip_range(args.humanml3d, keyid, tmr_annotations)
            motions.append(motion_loader(path=tmr_annotations[keyid]["path"], start=start, end=end))
        motion_emb = encode(motions)
        text_emb = encode(text_to_token_emb([p["text"] for p in pairs]))

    np.savez(args.out, ids=np.array(ids), text=text_emb, motion=motion_emb,
             similarity=np.array("cosine"))
    print(f"saved {len(ids)} pairs to {args.out}")


if __name__ == "__main__":
    main()
