# StuCrsフレームワークのパフォーマンスを計る試験データ

## 実行環境
- CPU : intel corei5 12400
- メモリ : 48GB
- GPU : NVIDIA Geforce RTX 4060  VRAM 8GB

## CPU環境における学習パフォーマンス
ニューラルネットワークの種類は**Example** にあるコードです。
- ① : [mnist_shallow_nn.rs](examples/mnist_shallow_nn.rs)
- ② : [mnist_deep_nn.rs](examples/mnist_deep_nn.rs)
- ③ : [mnist_CNN1.rs](examples/mnist_CNN1.rs)


| ニューラルネットワークの種類 /フレームワーク|StuCrs|Dezero|TensorFlow|
|---------------------------|---|---|---|
|①のニューラルネットワーク|21.25811s | 35.70813s| 17.60617s|
|②''|49.17341s|76.68909s|35.39685s|
|③''|3779.9s|なし|378.3298s|
