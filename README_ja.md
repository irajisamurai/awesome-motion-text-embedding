# Awesome Motion-Text Embedding

[English](README.md) | **日本語**

3D人体モーションと自然言語の共通埋め込み空間を学習するモーション-テキスト埋め込みモデルと、その学習・評価に使われるデータセットをまとめたリストです。

## 目次

- [手法](#手法)
- [データセット](#データセット)
- [リーダーボード](#リーダーボード)
- [評価](#評価)

## 手法

| 手法 | 論文 | 発表先 | リンク | コード |
|---|---|---|---|---|
| Guo et al. | Generating Diverse and Natural 3D Human Motions from Text | CVPR 2022 | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html) | [GitHub](https://github.com/EricGuo5513/text-to-motion) |
| MotionCLIP | MotionCLIP: Exposing Human Motion Generation to CLIP Space | ECCV 2022 | [arXiv](https://arxiv.org/abs/2203.08063) | [GitHub](https://github.com/GuyTevet/MotionCLIP) |
| TEMOS | TEMOS: Generating diverse human motions from textual descriptions | ECCV 2022 | [arXiv](https://arxiv.org/abs/2204.14109) | [GitHub](https://github.com/Mathux/TEMOS) |
| TMR | TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis | ICCV 2023 | [arXiv](https://arxiv.org/abs/2305.00976) | [GitHub](https://github.com/Mathux/TMR) |
| MoPatch | Exploring Vision Transformers for 3D Human Motion-Language Models with Motion Patches | CVPR 2024 | [arXiv](https://arxiv.org/abs/2405.04771) | [GitHub](https://github.com/line/MotionPatches) |
| LAVIMO | Tri-Modal Motion Retrieval by Learning a Joint Embedding Space | CVPR 2024 | [arXiv](https://arxiv.org/abs/2403.00691) | 非公開 |
| LaMP | LaMP: Language-Motion Pretraining for Motion Generation, Retrieval, and Captioning | ICLR 2025 | [arXiv](https://arxiv.org/abs/2410.07093) | [GitHub](https://github.com/gentlefress/LaMP) |
| SGAR | SGAR: Structural Generative Augmentation for 3D Human Motion Retrieval | NeurIPS 2025 | [Paper](https://openreview.net/pdf/1d98cebdbc17f9dacf4861cfbd15a16f71dd3e8a.pdf) / [Project](https://jhang2020.github.io/Projects/SGAR/SGAR.html) | [GitHub](https://github.com/JHang2020/SGAR) |
| MotionBind | MotionBind: Multi-Modal Human Motion Alignment for Retrieval, Recognition, and Generation | NeurIPS 2025 | [OpenReview](https://openreview.net/forum?id=sUjwDdyspc) / [Project](https://vidal-lab.github.io/MotionBind/) | 非公開 |

## データセット

| データセット | 論文 | 発表先 | リンク |
|---|---|---|---|
| KIT-ML | The KIT Motion-Language Dataset | Big Data 2016 | [arXiv](https://arxiv.org/abs/1607.03827) |
| HumanML3D | Generating Diverse and Natural 3D Human Motions from Text | CVPR 2022 | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html) / [GitHub](https://github.com/EricGuo5513/HumanML3D) |

## リーダーボード

すべての数値は、固定した HumanML3D のテストセットと共通の評価スクリプトを使って、このリポジトリで測り直したものです（[評価](#評価)を参照）。そのため、論文に書かれた数値とは異なる場合があります。2〜7列目が Text-to-Motion、8〜13列目が Motion-to-Text の検索結果です。

### HumanML3D（All protocol）

| 手法 | R@1 | R@2 | R@3 | R@5 | R@10 | MedR | R@1 | R@2 | R@3 | R@5 | R@10 | MedR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Guo et al. | 1.85 | 3.49 | 4.84 | 7.19 | 12.57 | 81.00 | 2.40 | 3.70 | 5.41 | 8.01 | 12.91 | 81.00 |

### HumanML3D（Small-Batch protocol）

| 手法 | R@1 | R@2 | R@3 | R@5 | R@10 | MedR | R@1 | R@2 | R@3 | R@5 | R@10 | MedR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Guo et al. | 51.16 | 71.67 | 81.07 | 89.92 | 96.49 | 1.46 | 50.68 | 71.90 | 81.73 | 90.15 | 97.06 | 1.51 |

### 再現方法

手法ごとにスクリプトを用意しています。スクリプトは、公式コードを決まったコミットで clone し、学習済みモデルをダウンロードして、テストペアの埋め込みを出力し、指標を計算します。各手法の Python 環境と HumanML3D は各自で用意して、次のコマンドを実行してください。

| 手法 | コマンド | 結果 |
|---|---|---|
| Guo et al. | `bash leaderboard/methods/guo/run.sh /path/to/HumanML3D` | [guo.json](leaderboard/results/guo.json) |

## 評価

モーション-テキスト埋め込みモデルは、各データセットのテスト分割を使ったクロスモーダル検索で評価します。テストサンプルはそれぞれ（モーション, テキスト）のペアで、検索対象に含まれるすべてのテキスト埋め込みとモーション埋め込みの間で類似度行列を計算します。

### テストセット

HumanML3D のテストセットは、4,384 個の（モーション, テキスト）ペアに固定します。一覧は [`leaderboard/benchmark/humanml3d_test_pairs.json`](leaderboard/benchmark/humanml3d_test_pairs.json) にあります。

| 項目 | 定義 |
|---|---|
| サンプル | HumanML3D の `test.txt` にある 4,384 個の ID すべて（元のモーション 2,192 個と、`M` で始まる左右反転版 2,192 個）。 |
| モーション | 各 ID のクリップ全体。テキストに付いている開始・終了時刻では切り出しません。モーションの表現形式や長さの扱いは各手法に任せます。 |
| テキスト | `texts/<id>.txt` の1行目の、最初の `#` より前の文。 |

### 評価指標

評価指標は両方向で報告します。

- **Text-to-Motion (T2M)**：テキストをクエリとして、モーションを順位付けします。
- **Motion-to-Text (M2T)**：モーションをクエリとして、テキストを順位付けします。

| 指標 | 説明 | 良い方向 |
|---|---|---|
| R@1, R@2, R@3, R@5, R@10 | Recall at K：正解が上位K件に入ったクエリの割合（%）。 | 高いほど良い |
| MedR | 全クエリにおける正解の順位の中央値（1位が最上位）。 | 低いほど良い |

### 評価プロトコル

どちらのプロトコルでも、正解はペアになっているサンプルだけです。違いは検索対象の大きさです。

| プロトコル | 検索対象 |
|---|---|
| Small-Batch | 32件ずつのランダムなバッチ。指標はバッチ全体で平均します。 |
| All | テストセット全体。 |

- **Small-Batch protocol** は [Guo et al.](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html) の評価方法に従います。ID をソートしてから `np.arange(4384)` を `np.random.RandomState(0)` でシャッフルし、32件ずつ137個のバッチに分けて、各バッチの中で検索します。バッチの一覧は、テストペアの JSON の `small_batches` にあります。1つのクエリに対する不正解の候補が31件しかないため、All protocol よりもスコアがかなり高くなります。
- **All protocol** はテストセット全体から検索します。正解と同じスコアの候補が複数ある場合は、それらの順位を平均します。

プロトコルが違う結果は比較できないので、論文がどのプロトコルで報告しているかを必ず確認してください。
