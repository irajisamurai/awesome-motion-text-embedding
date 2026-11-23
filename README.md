# Awesome Motion-Text Embedding

**English** | [日本語](README_ja.md)

A curated list of motion-text embedding models, which learn a joint space for 3D human motion and natural language, and the datasets used to train and evaluate them.

## Contents

- [Methods](#methods)
- [Datasets](#datasets)
- [Leaderboard](#leaderboard)
- [Evaluation](#evaluation)

## Methods

| Method | Paper | Venue | Links | Code |
|---|---|---|---|---|
| Guo et al. | Generating Diverse and Natural 3D Human Motions from Text | CVPR 2022 | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html) | [GitHub](https://github.com/EricGuo5513/text-to-motion) |
| MotionCLIP | MotionCLIP: Exposing Human Motion Generation to CLIP Space | ECCV 2022 | [arXiv](https://arxiv.org/abs/2203.08063) | [GitHub](https://github.com/GuyTevet/MotionCLIP) |
| TEMOS | TEMOS: Generating diverse human motions from textual descriptions | ECCV 2022 | [arXiv](https://arxiv.org/abs/2204.14109) | [GitHub](https://github.com/Mathux/TEMOS) |
| TMR | TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis | ICCV 2023 | [arXiv](https://arxiv.org/abs/2305.00976) | [GitHub](https://github.com/Mathux/TMR) |
| MoPatch | Exploring Vision Transformers for 3D Human Motion-Language Models with Motion Patches | CVPR 2024 | [arXiv](https://arxiv.org/abs/2405.04771) | [GitHub](https://github.com/line/MotionPatches) |
| LAVIMO | Tri-Modal Motion Retrieval by Learning a Joint Embedding Space | CVPR 2024 | [arXiv](https://arxiv.org/abs/2403.00691) | Not released |
| LaMP | LaMP: Language-Motion Pretraining for Motion Generation, Retrieval, and Captioning | ICLR 2025 | [arXiv](https://arxiv.org/abs/2410.07093) | [GitHub](https://github.com/gentlefress/LaMP) |
| SGAR | SGAR: Structural Generative Augmentation for 3D Human Motion Retrieval | NeurIPS 2025 | [Paper](https://openreview.net/pdf/1d98cebdbc17f9dacf4861cfbd15a16f71dd3e8a.pdf) / [Project](https://jhang2020.github.io/Projects/SGAR/SGAR.html) | [GitHub](https://github.com/JHang2020/SGAR) |
| MotionBind | MotionBind: Multi-Modal Human Motion Alignment for Retrieval, Recognition, and Generation | NeurIPS 2025 | [OpenReview](https://openreview.net/forum?id=sUjwDdyspc) / [Project](https://vidal-lab.github.io/MotionBind/) | Not released |

## Datasets

| Dataset | Paper | Venue | Links |
|---|---|---|---|
| KIT-ML | The KIT Motion-Language Dataset | Big Data 2016 | [arXiv](https://arxiv.org/abs/1607.03827) |
| HumanML3D | Generating Diverse and Natural 3D Human Motions from Text | CVPR 2022 | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html) / [GitHub](https://github.com/EricGuo5513/HumanML3D) |

## Leaderboard

All results are measured by us on the fixed HumanML3D test set with the same evaluation script (see [Evaluation](#evaluation)), so they may differ from the numbers reported in the papers. Columns 2-7 are Text-to-Motion and columns 8-13 are Motion-to-Text retrieval.

### HumanML3D (All protocol)

| Method | R@1 | R@2 | R@3 | R@5 | R@10 | MedR | R@1 | R@2 | R@3 | R@5 | R@10 | MedR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Guo et al. | 1.85 | 3.49 | 4.84 | 7.19 | 12.57 | 81.00 | 2.94 | 3.76 | 6.04 | 8.44 | 12.96 | 81.00 |
| TMR | 5.89 | 10.93 | 14.26 | 20.53 | 31.41 | 28.00 | 9.15 | 11.66 | 16.79 | 22.63 | 32.46 | 26.00 |
| MoPatch | 7.28 | 13.75 | 18.45 | 25.66 | 38.12 | 19.00 | 11.91 | 14.83 | 21.26 | 28.26 | 38.75 | 19.00 |
| LaMP† | 4.01 | 7.94 | 10.72 | 15.99 | 26.25 | 33.00 | 6.30 | 7.94 | 13.55 | 19.21 | 28.12 | 31.50 |
| SGAR | 8.71 | 15.21 | 19.37 | 27.62 | 40.51 | 17.00 | 13.30 | 16.72 | 23.68 | 30.52 | 41.20 | 16.00 |

### HumanML3D (Small-Batch protocol)

| Method | R@1 | R@2 | R@3 | R@5 | R@10 | MedR | R@1 | R@2 | R@3 | R@5 | R@10 | MedR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Guo et al. | 51.16 | 71.67 | 81.07 | 89.92 | 96.49 | 1.46 | 50.78 | 71.94 | 81.73 | 90.15 | 97.06 | 1.51 |
| TMR | 67.38 | 81.07 | 86.18 | 91.40 | 95.39 | 1.03 | 68.11 | 81.87 | 86.75 | 91.17 | 95.42 | 1.03 |
| MoPatch | 73.04 | 85.77 | 89.85 | 93.41 | 96.83 | 1.00 | 72.67 | 86.27 | 90.24 | 93.68 | 96.85 | 1.00 |
| LaMP† | 66.22 | 81.27 | 86.06 | 90.17 | 92.77 | 1.02 | 66.38 | 80.93 | 86.25 | 90.08 | 92.81 | 1.03 |
| SGAR | 75.41 | 88.07 | 92.34 | 95.78 | 97.79 | 1.00 | 75.73 | 88.50 | 92.45 | 95.53 | 98.11 | 1.00 |

### Reproduce

Each method has a script that clones the official code at a fixed commit, downloads the pretrained model, exports the embeddings of the test pairs, and computes the metrics. Prepare the Python environment of the method and HumanML3D yourself, then run:

| Method | Command | Result |
|---|---|---|
| Guo et al. | `bash leaderboard/methods/guo/run.sh /path/to/HumanML3D` | [guo.json](leaderboard/results/guo.json) |
| TMR | `bash leaderboard/methods/tmr/run.sh /path/to/HumanML3D /path/to/pose_data` | [tmr.json](leaderboard/results/tmr.json) |
| MoPatch | `bash leaderboard/methods/mopatch/run.sh /path/to/HumanML3D` | [mopatch.json](leaderboard/results/mopatch.json) |
| LaMP† | `bash leaderboard/methods/lamp/run.sh /path/to/HumanML3D` | [lamp.json](leaderboard/results/lamp.json) |
| SGAR | `bash leaderboard/methods/sgar/run.sh /path/to/HumanML3D` | [sgar.json](leaderboard/results/sgar.json) |

† The LaMP checkpoint does not include the normalization statistics used in training. We normalize the motions with the evaluator statistics of Guo et al., which the released LaMP evaluation code also uses and which reproduce the small-batch results of the paper (T2M R@1 66.22 vs 67.18). See [`leaderboard/methods/lamp/export_embeddings.py`](leaderboard/methods/lamp/export_embeddings.py).

## Evaluation

Motion-text embedding models are evaluated by cross-modal retrieval on the test split of each dataset. Each test sample is a (motion, text) pair, and a similarity matrix is computed between all text and motion embeddings in the evaluation pool.

### Test set

The HumanML3D test set is fixed as 4,384 (motion, text) pairs. The list is in [`leaderboard/benchmark/humanml3d_test_pairs.json`](leaderboard/benchmark/humanml3d_test_pairs.json).

| Item | Definition |
|---|---|
| Samples | All 4,384 IDs in HumanML3D `test.txt` (2,192 original motions and their 2,192 mirrored versions, whose IDs start with `M`). |
| Motion | The whole clip of each ID. Start/end tags in the captions are not used to crop the motion. The motion representation and length handling are up to each method. |
| Text | The first line of `texts/<id>.txt`, using the caption before the first `#`. |

Pairs with identical captions share one text embedding (that of the first occurrence), so identical captions always get exactly the same scores and their ties are handled the same way on every machine.

### Metrics

Metrics are reported in both directions:

- **Text-to-Motion (T2M)**: a text is the query, and motions are ranked.
- **Motion-to-Text (M2T)**: a motion is the query, and texts are ranked.

| Metric | Description | Better |
|---|---|---|
| R@1, R@2, R@3, R@5, R@10 | Recall at K: the percentage of queries whose correct match is ranked in the top K. | Higher |
| MedR | Median rank of the correct match over all queries (rank 1 is the top). | Lower |

### Protocols

In both protocols, only the paired sample counts as the correct match. The protocols differ in the size of the evaluation pool.

| Protocol | Evaluation pool |
|---|---|
| Small-Batch | Random batches of 32 samples; metrics are averaged over the batches. |
| All | The whole test set. |

- **Small-Batch protocol** follows the evaluation of [Guo et al.](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html). The IDs are sorted, `np.arange(4384)` is shuffled with `np.random.RandomState(0)`, and the result is cut into 137 batches of 32. Retrieval is run inside each batch. The batches are listed in `small_batches` of the test-pair JSON. Because there are only 31 distractors per query, scores are much higher than with the All protocol.
- **All protocol** retrieves from the whole test set. When several candidates have the same score as the correct match, their ranks are averaged.

Results under different protocols are not comparable, so always check which protocol a paper reports.
