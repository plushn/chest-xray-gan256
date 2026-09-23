# GAN_dicom256

胸部単純写真（DICOM）を学習データとして、GAN（Generative Adversarial Network）で
256×256のグレースケール胸部X線画像を生成する学習用プロジェクトです。

数年前に「GANの勉強」として書いたコードで、以下の2種類の実装が含まれます。

- **DCGAN** ([dcgan.py](dcgan.py)) : 標準的なDCGAN
- **SGAN** ([sgan.py](sgan.py) / [sgan_pt.py](sgan_pt.py)) : 結節（nodule）/ 非結節（non-nodule）の2クラスラベルを使った半教師あり学習GAN（Semi-Supervised GAN）
  - `sgan.py` : TensorFlow/Keras実装
  - `sgan_pt.py` : PyTorchへの移植版（未完成・参考実装）

> **Note**
> 学習用に書いたコードをそのまま公開しています。リファクタリングは行っておらず、
> コメントアウトされた実験コードや未整理の部分が残っています。

## 構成

```
.
├── dcgan.py               # DCGAN本体（学習・画像生成）
├── sgan.py                # SGAN本体（TensorFlow/Keras版）
├── sgan_pt.py              # SGAN（PyTorch移植版, 参考実装）
├── gen_data.py             # DCGAN用データ前処理（DICOM → numpy）
├── gen_data_sgan_v2.py      # SGAN用データ前処理（DICOM → numpy, ラベル付与・train/test分割）
├── gen_movies.py           # 学習過程の生成画像からmp4アニメーションを作成
├── import unittest.py      # sgan.py の Generator 構造に対する簡易ユニットテスト
├── requirements.txt
└── readme.txt              # 当時の検討メモ（研究計画のドラフト）
```

DICOM原データ・前処理済みnumpyデータ（`*.npy`）・学習済み重み・生成画像・ログ・動画は
サイズが大きいことと医療画像データであることから、このリポジトリには含めていません
（`.gitignore` で除外）。手元で `DICOMdata/`（胸部X線DICOM一式）、
`DICOMdata_Nodule/` / `DICOMdata_NonNodule/`（結節あり/なしのDICOM）を用意して実行してください。

## セットアップ

```bash
pip install -r requirements.txt
```

`sgan.py` / `dcgan.py` は `keras.layers` を直接importする古い書き方（TensorFlow 2.9〜2.15系の
バンドルKeras想定）になっているため、最新のTensorFlow/Keras（Keras 3系）では
そのまま動かない可能性があります。

## 使い方

### 1. データ前処理

DICOMファイルを256×256に正規化してnumpy配列として保存します。

```bash
# DCGAN用: ./DICOMdata/*.dcm を読み込み gen_data1.npy を生成
python gen_data.py

# SGAN用: ./DICOMdata_Nodule, ./DICOMdata_NonNodule を読み込み
# ラベル付き gen_data_sgan1.npy (train/test分割済み) を生成
python gen_data_sgan_v2.py
```

### 2. 学習

```bash
# DCGAN（gen_data1.npy を使用）
python dcgan.py

# SGAN（gen_data_sgan1.npy を使用）
python sgan.py
```

学習中、一定エポックごとに生成画像を `generated_images/` (DCGAN) や
`generated_images_sgan/` (SGAN) に保存し、損失（loss）を `loss.csv` / `loss_sgan.csv` に
追記します。

### 3. 生成過程の動画化（任意）

```bash
python gen_movies.py
```

`generated_images/` 内の画像を連結して `animation.mp4` を作成します。

## モデル概要

- **Generator** : 潜在変数（100次元）を全結合層で8×8×8の特徴マップに変換し、
  `UpSampling2D` + `Conv2D` を5段重ねて256×256まで拡大（U-Net的なエンコーダ〜デコーダの
  スキップ接続は持たない、単純なdecoderのみの構成）
- **Discriminator** : `Conv2D`（stride=2）を6段重ねて4×4まで縮小し、`Dense`で真偽判定
- **SGAN** はDiscriminatorを教師あり（結節/非結節の2クラス分類）と教師なし（真偽判定）で
  共有し、半教師あり学習を行う構成

## 動作環境（当時）

- Python 3.9系
- TensorFlow / Keras（2.9〜2.15系を想定）
- Apple Silicon Mac (`mps` バックエンド利用コードが一部に残っています)

## ライセンス / データについて

コードは学習目的で公開しています。学習に使用したDICOMデータ自体はこのリポジトリに
含まれていません。利用するデータセットの利用規約・ライセンスに従ってください。
