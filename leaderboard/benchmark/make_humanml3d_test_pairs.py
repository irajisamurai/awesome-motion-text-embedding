"""Write the HumanML3D test pairs used by the leaderboard to JSON.

Usage:
    python leaderboard/benchmark/make_humanml3d_test_pairs.py \
        /path/to/HumanML3D leaderboard/benchmark/humanml3d_test_pairs.json

/path/to/HumanML3D must contain test.txt and texts/.
"""
import json, os, sys
import numpy as np

hml, out = sys.argv[1], sys.argv[2]
ids = open(os.path.join(hml, "test.txt")).read().strip().split("\n")
assert len(ids) == len(set(ids)) == 4384

pairs = []
for i in ids:
    first = open(os.path.join(hml, "texts", i + ".txt"), encoding="utf-8").read().strip().split("\n")[0]
    text = first.split("#")[0].strip()
    assert text, i
    pairs.append({"id": i, "text": text})

keyids = sorted(ids)
idx = np.arange(len(keyids))
np.random.RandomState(0).shuffle(idx)
batches = [[keyids[j] for j in idx[32 * b: 32 * (b + 1)]] for b in range(len(keyids) // 32)]
assert sum(map(len, batches)) == 4384 and all(len(b) == 32 for b in batches)

data = {
    "dataset": "HumanML3D",
    "split": "test",
    "num_pairs": len(pairs),
    "definition": {
        "samples": "All IDs in HumanML3D test.txt (2192 original + 2192 mirrored 'M' IDs).",
        "motion": "The whole clip of each ID; captions with start/end tags are NOT used to crop the motion.",
        "text": "The first line of texts/<id>.txt, the part before the first '#', stripped.",
        "all_protocol": "Retrieval over all pairs; the paired sample is the only correct match.",
        "small_batch_protocol": "Sort the IDs, shuffle np.arange(N) with np.random.RandomState(0), cut into consecutive chunks of 32; metrics are averaged over the batches.",
    },
    "pairs": pairs,
    "small_batches": batches,
}
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=1)
print("pairs", len(pairs), "batches", len(batches))
