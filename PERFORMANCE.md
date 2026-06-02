# StuCrsフレームワークのパフォーマンスを計る試験データ

## 実行環境
- CPU : intel corei5 12400
- メモリ : 48GB
- GPU : NVIDIA Geforce RTX 4060  VRAM 8GB

## 注意
これらの処理速度はRustの標準ライブラリである`std::time::Instant`を用いて計測しています。計測時間はあくまで目安として参考にしてください。


## Version 1

ニューラルネットワークの種類は**Example** にあるコードです。
- ① : [mnist_shallow_nn.rs](examples/v1/mnist_shallow_nn.rs)
- ② : [mnist_deep_nn.rs](examples/v1/mnist_deep_nn.rs)
- ③ : [mnist_CNN1.rs](examples/v1/mnist_CNN1.rs)


### CPU環境における学習速度


| ニューラルネットワークの種類 /フレームワーク|StuCrs|Dezero|TensorFlow|
|---------------------------|---|---|---|
|①のニューラルネットワーク|21.25811s | 35.70813s| 17.60617s|
|②''|49.17341s|76.68909s|35.39685s|
|③''|3779.9s|しゃ




## Version 2

ニューラルネットワークの種類は**Example** にあるコードです。
- ① : [mnist_shallow_nn.rs](examples/v2/mnist_shallow_nn.rs)
- ② : [mnist_deep_nn.rs](examples/v2/mnist_deep_nn.rs)
- ③ : [mnist_shallow_cnn.rs](examples/v2/mnist_shallow_cnn_v2.rs)


---

### V2.1 (自作カーソル)

### CPU環境における学習速度


| ニューラルネットワークの種類 /フレームワーク|StuCrs|Dezero|TensorFlow|
|---------------------------|---|---|---|
|①のニューラルネットワーク|21.25811s | 35.70813s| 17.60617s|
|②''|49.17341s|76.68909s|35.39685s|
|③''|3779.9s|なし|378.3298s|



### GPU (CUDA)環境における学習速度

| ニューラルネットワークの種類 /バージョン|V2.1|V2.2|
|---------------------------|---|---|
|①のニューラルネットワーク|なし| 5.72778s|
|②のニューラルネットワーク|なし|7.18018s|  
|③のニューラルネットワーク : batch = 128|なし|15.1082s|
|③のニューラルネットワーク : batch = 512|13.6312s|12.6265s|
|③のニューラルネットワーク : batch = 1024|13.4769s|12.4636s|

