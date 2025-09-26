
## 研究概要

本研究では Rust言語を用いて「StuCrs」というディープラーニングのフレームワークを一から実装、開発しました。StuCrsというフレームワークの特徴はフルRust実装で、直感的に原理を理解できるシンプルな構造となっており、
ユーザーが一から実装し、深層学習の原理の理解を深めてもらう教材としての役割を果たすフレームワークです。また、Rust言語を学びたい方にとっても良いサンプルコードです。

## 背景
・TensorFlowやPyTorchといった既存のフレームワークのほとんどがドキュメントやコミュニティが英語だったりと、日本語によるフレームワークの開発にとって障壁となっている。

・機械学習の開発がpythonやC系言語に比べてRustは遅れている。


## 研究のコンセプト
・日本語によってコードの説明をすることでユーザー自らが一からフレームワークを実装してもらい、深層学習の原理を探究してもらうこと。

・日本語のコミュニティを構築しやすい国産のフレームワークを、機械学習で開発途上のRustで実装することで、さらなる日本でのRustにおける深層学習のコミュニティを活発にし、開発を促すこと。

## 研究にあたって
本研究は下の著書『ゼロから作るDeep Learning③フレームワーク編』をもとにして実装しています。著者である斎藤康毅氏に著書の考えや表現の使用を許可していただいたことに感謝を申し上げるとともに、この著書オリジナルのフレームワークDeZeroも研究の参考として利用させていただいています。
<p><img width="280" height="134" alt="Image" src="https://github.com/user-attachments/assets/6c0ddf88-3371-40aa-a131-075947068e1b" /> &emsp;
  <img width="100" height="142" alt="Image" src="https://github.com/user-attachments/assets/d5d1ca74-79cb-4de3-b55c-537c705788f7" />


## ドキュメント


開発した深層学習のフレームワーク「StuCrs」の実装までのコードの説明をこちらのドキュメントで見ることができます。これを読んでぜひ一からRustでフレームワークを実装してみましょう！
<https://docs.google.com/document/d/1jJL_ijYnqIFADSTfTqLcnNre754g24bE963L_r3hwus/edit?usp=sharing>


## ファイル構成

|フォルダ名 |説明         |
|:--        |:--                  |
|[stucrs](/stucrs)       |StuCrs(CPU用)のソースコード|
|[stucrs-gpu](/stucrs)    |StuCrs(GPU用)のソースコード|
|[assets](/cuda_test)     |StuCrsを用いて様々な実験した際のデータや画像|



## 使用した外部のクレート

本研究で必要とする外部クレートとバージョンは下記の通りです。

- [ndarray-0.16.0](https://docs.rs/ndarray/0.16.0/ndarray/index.html)


NVIDIAのGPUで実行できる機能も提供しています。その場合はstucrs-gpuをダウンロードし、また下記のtensor_frameクレートを使用します。

- [tensor_frame](https://docs.rs/tensor_frame/latest/tensor_frame/index.html) （オプション）


## 実行方法

フォルダーのstucrsをダウンロードしていただき、外部クレートとしてご利用ください。また、こちらのクレートはバグといった不具合の対応が不十分だと判断し、ライブラリクレートとしては公開しておりません。またオプションとして、NVIDIAのGPUで実行できる機能も提供しています。その場合はstucrs-gpuをダウンロードしてください。


```
cargo run --release
```


## 試験データ

StuCrsを実際に実行して処理速度などを計測した試験データをこちらに公開しております。　<https://docs.google.com/spreadsheets/d/1Fkxn7yqLILJlHYeADVa_jJljFBYD5ZvIH0I7-EVTuFU/edit?usp=sharing>


## 最後に
はじめに、私たちの研究に目を通していただきありがとうございます。
本研究は素人である高校生が独自に研究したものであり、Rustのパフォーマンス的に、もしくは習わし的にふさわしくないコード、また深層学習の知識における間違いが多くあるかと思います。もし気になる点や改善した方がいいというご意見がございましたら、是非ともお手柔らかにお知らせください。たくさんのご意見、ご感想をお待ちしております。
