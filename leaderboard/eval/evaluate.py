"""Compute the leaderboard metrics from exported embeddings.

Usage:
    python leaderboard/eval/evaluate.py EMBEDDINGS.npz OUTPUT.json \
        [--pairs leaderboard/benchmark/humanml3d_test_pairs.json]

EMBEDDINGS.npz must contain:
    ids         [N]     sample IDs (any order; every ID in the pair list must be present once)
    text        [N, D]  text embeddings
    motion      [N, D]  motion embeddings (row i is paired with text row i)
    similarity  str     "cosine" (embeddings are L2-normalized here) or
                        "neg_euclidean" (similarity = -Euclidean distance)

Metrics follow TMR / SGAR (scripts/metrics.py): the paired sample is the only correct
match, ties are broken by averaging the ranks, R@K is a percentage and MedR starts at 1.
"""
import argparse
import json
import os

import numpy as np

RECALL_KS = (1, 2, 3, 5, 10)
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PAIRS = os.path.join(HERE, "..", "benchmark", "humanml3d_test_pairs.json")


def similarity_matrix(text, motion, kind):
    text = text.astype(np.float64)
    motion = motion.astype(np.float64)
    if kind == "cosine":
        text = text / np.linalg.norm(text, axis=1, keepdims=True)
        motion = motion / np.linalg.norm(motion, axis=1, keepdims=True)
        return text @ motion.T
    if kind == "neg_euclidean":
        d2 = (text ** 2).sum(1)[:, None] + (motion ** 2).sum(1)[None, :] - 2 * text @ motion.T
        return -np.sqrt(np.maximum(d2, 0.0))
    raise ValueError(f"unknown similarity: {kind}")


def contrastive_metrics(sims):
    """sims: [N, N], rows are queries, the correct match is on the diagonal."""
    n, m = sims.shape
    assert n == m
    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    gt_dists = np.diag(dists)[:, None]
    locs = np.argwhere((sorted_dists - gt_dists) == 0)
    if locs.shape[0] > n:  # ties: average rank
        splits = np.insert(np.nonzero(np.diff(locs[:, 0]))[0] + 1, 0, 0)
        cols = np.add.reduceat(locs[:, 1], splits) / np.diff(np.append(splits, locs.shape[0]))
    else:
        cols = locs[:, 1]
    assert cols.size == n, f"expected ranks to match queries ({cols.size} vs {n})"
    out = {f"R{k:02d}": 100.0 * float(np.sum(cols < k)) / n for k in RECALL_KS}
    out["MedR"] = float(np.median(cols) + 1)
    return out


def both_directions(sims):
    out = {}
    for name, s in (("t2m", sims), ("m2t", sims.T)):
        for k, v in contrastive_metrics(s).items():
            out[f"{name}/{k}"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("embeddings")
    ap.add_argument("output")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    args = ap.parse_args()

    bench = json.load(open(args.pairs, encoding="utf-8"))
    order = [p["id"] for p in bench["pairs"]]

    emb = np.load(args.embeddings, allow_pickle=False)
    ids = [str(i) for i in emb["ids"]]
    kind = str(emb["similarity"])
    assert len(ids) == len(set(ids)), "duplicate IDs in embeddings"
    missing = set(order) - set(ids)
    assert not missing, f"{len(missing)} benchmark IDs are missing, e.g. {sorted(missing)[:5]}"
    pos = {k: i for i, k in enumerate(ids)}
    sel = np.array([pos[k] for k in order])
    sims = similarity_matrix(emb["text"][sel], emb["motion"][sel], kind)

    index = {k: i for i, k in enumerate(order)}
    results = {"all": both_directions(sims)}
    per = [both_directions(sims[np.ix_(b, b)])
           for b in ([index[k] for k in batch] for batch in bench["small_batches"])]
    results["small_batch"] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    for proto in results.values():
        for k in proto:
            proto[k] = round(proto[k], 2)

    out = {
        "dataset": bench["dataset"],
        "num_pairs": len(order),
        "num_small_batches": len(bench["small_batches"]),
        "similarity": kind,
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
