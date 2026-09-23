#1 gen_data.py
    dicomデータを読み込み、画像を生成する
    256x256にリサイズし、0-255の範囲に正規化する。それをnumpy配列として保存する。

#2 gen_data.py

1. Introduction
2. Methods
    1. Data
        1. Data source:桜山および東部医療センターでの撮影データを使用する。症例数はそれぞれ100症例を目標とする。
        2. Data preprocessing: 256x256にリサイズし、0-255の範囲に正規化する。
    2. Preprocessing
        1. Data augmentation
        2. Data split
    3. Model
        1. Model architecture
        2. Model training
    4. Training
    5. Evaluation
3. Results
4. Discussion
5. Conclusion