# Awesome Motion-Text Embedding

**English** | [日本語](README_ja.md)

A curated list of motion-text embedding models, which learn a joint space for 3D human motion and natural language, and the datasets used to train and evaluate them.

## Contents

- [Methods](#methods)
- [Datasets](#datasets)
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
| SGAR | SGAR: Structural Generative Augmentation for 3D Human Motion Retrieval | NeurIPS 2025 | [Paper](https://openreview.net/pdf/1d98cebdbc17f9dacf4861cfbd15a16f71dd3e8a.pdf) / [Project](https://jhang2020.github.io/Projects/SGAR/SGAR.html) | [GitHub](https://github.com/JHang2020/SGAR) |
| MotionBind | MotionBind: Multi-Modal Human Motion Alignment for Retrieval, Recognition, and Generation | NeurIPS 2025 | [OpenReview](https://openreview.net/forum?id=sUjwDdyspc) / [Project](https://vidal-lab.github.io/MotionBind/) | Not released |

## Datasets

| Dataset | Paper | Venue | Links |
|---|---|---|---|
| KIT-ML | The KIT Motion-Language Dataset | Big Data 2016 | [arXiv](https://arxiv.org/abs/1607.03827) |
| HumanML3D | Generating Diverse and Natural 3D Human Motions from Text | CVPR 2022 | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html) / [GitHub](https://github.com/EricGuo5513/HumanML3D) |

## Evaluation

Motion-text embedding models are evaluated by cross-modal retrieval on the test split of each dataset. Each test sample is a (motion, text) pair, and a similarity matrix is computed between all text and motion embeddings in the evaluation pool.

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

- **Small-Batch protocol** follows the evaluation of [Guo et al.](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html). The test set is shuffled with a fixed seed, split into batches of 32, and retrieval is run inside each batch. Because there are only 31 distractors per query, scores are much higher than with the All protocol.
- **All protocol** retrieves from the whole test set. When several candidates have the same score as the correct match, their ranks are averaged.

Results under different protocols are not comparable, so always check which protocol a paper reports.
