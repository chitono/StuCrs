# Im2col関数の実装

では理論的な説明が終わったということで、実装に移ります。はじめにフィルターの範囲のデータを取りだす **Im2col関数** を実装していきます。

先ほど畳み込みの処理を説明しましたが、少々複雑な処理だったと思います。それを実際に実装していくわけですが、行列自体を計算する関数と **Function構造体** の二つに分けて実装していきます。その前に一つだけ、共通して使用する重要な関数を実装しておきます。

## get_conv_outsize関数

こちらの関数は、畳み込み計算において、画像データのサイズ、ストライド、パディングのデータから、出力データのサイズを計算する関数です。この説明は[Conv2d関数の理論](../CNN_riron/cnn_riron_conv.md)のところで行った\\(OH,OW\\)を求めるものであり、そこでの計算式を利用します。     


```rust
pub fn get_conv_outsize(
    input_size: (usize, usize),
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> (usize, usize) {
    let oh = (input_size.0 + pad_size.0 * 2 - kernel_size.0) / stride_size.0 + 1;
    let ow = (input_size.1 + pad_size.1 * 2 - kernel_size.1) / stride_size.1 + 1;

    (oh, ow)
}
```

前回表示した計算式をそのまま使用したので特に問題はないのですが、ここで重要なのは、今後私たちが実装するCNNの関数や機能は、オプションとなるデータ(インプットサイズやストライドサイズなど)を基本的に長さ2の **タプル** で扱います。これは画像データが2次元であり、設定する値の多くが **(縦、横)** の2次元であるためです。この関数で簡単に出力の\\(OH,OW\\)を求めることができます。


使い方の確認と、計算のテストをします。
```rust
#[test]
    fn get_conv_outsize_test() {
        use crate::functions_cnn::get_conv_outsize;

        let input_size = (4, 4);
        let kernel_size = (3, 3);
        let stride_size = (1, 1);
        let pad_size = (1, 1);

        let output_size = get_conv_outsize(input_size, kernel_size, stride_size, pad_size);

        assert_eq!(output_size, (4, 4));
    }
```

それではまず行列自体における計算についてです。

## Im2colの行列計算

Im2colという関数の名前は、入力データである **Image** を、フィルターで抽出できるような特殊な行列 **col** に変換する関数というのが由来となっています。

画像データから、入力されたフィルターのサイズやストライド、パディングをもとに、正しくフィルターの領域を順番に取り出す必要があります。この処理は行列のインデックスなどの情報を用いて非常に複雑に計算します。では実際に処理するコードをはじめに表示します。   

```rust
pub fn im2col_array(
    input: ArrayView4<f32>,
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> Array3<f32> {
    let input_shape = input.shape();

    // inputから形状のデータを取り出す。
    let n = input_shape[0]; //バッチ数
    let c = input_shape[1]; //チャンネル数
    let h = input_shape[2]; //縦
    let w = input_shape[3]; //横

    let (kh, kw) = kernel_size;
    let (stride_h, stride_w) = stride_size;
    let (pad_h, pad_w) = pad_size;

    let (oh, ow) = get_conv_outsize((h, w), (kh, kw), (stride_h, stride_w), (pad_h, pad_w));

    let mut cols = Array3::<f32>::zeros((n, c * kh * kw, oh * ow));  // colsを初期化 

    for b in 0..n {   // バッチから一枚の画像データを取り出す
        let img = input.slice(s![b, .., .., ..]);
        let mut col = cols.slice_mut(s![b, .., ..]);
        let mut col_idx = 0;

        for y in 0..oh {
            for x in 0..ow {
                let y_start = y as isize * stride_h as isize - pad_h as isize; 
                let x_start = x as isize * stride_w as isize - pad_w as isize;

                let mut patch = Vec::<f32>::with_capacity(c * kh * kw);

                for c_idx in 0..c {  
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let in_y = y_start + ky as isize;
                            let in_x = x_start + kx as isize;

                            // paddingしたところは0にする。
                            let value = if in_y >= 0
                                && (in_y as usize) < h
                                && in_x >= 0
                                && (in_x as usize) < w　// 参照した位置が画像の外か判別
                            {
                                img[(c_idx, in_y as usize, in_x as usize)]
                            } else {
                                0.0 // 元の画像データの範囲外なら0とする
                            };
                            patch.push(value);
                        }
                    }
                }
                for (i, v) in patch.into_iter().enumerate() {
                    col[(i, col_idx)] = v;
                }
                col_idx += 1;
            }
        }
    }
    cols
}
```



[Conv2d関数の理論](../CNN_riron/cnn_riron_conv.md)の **Im2col** のところで説明したように、入力データを3次元の行列に変換します。この時、3次元の行列にあたるのが、このコードで登場する **cols** です。colsを初期化するところを見ると、行列の形状が説明と同じだとわかります。この処理はループを非常に多く用いていて複雑に思えるかもしれませんが、Im2colの原理をしっかり理解している方ならこのコードを簡単に理解できると思います。最初にストライドとパディングを考慮してインデックスを生成し、そのインデックスのデータを参照して新たな行列にデータをコピーしていきます。この処理で重要な点はやはり **パディングの扱い** です。途中で、多くの条件で分離していますが、これはフィルター内のインデックスを参照しており、その時、そのインデックスがもとの画像データの外側にあるかを判別しているのです。外側にあれば、パディングの0でデータは0として扱われます。この処理で、0を周りに追加した行列を新たに生成していないことはおわかりいただけたでしょうか。それは仮想的に0というデータを用いることで、無駄に行列を生成することなく済むのです。

では先ほどと同じように計算の確認をします。
```rust
#[test]
    fn im2col_test() {
        use crate::functions_cnn::im2col_array;

        let input = array![[[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]]];
        let kernel_size = (2, 2);
        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let output = im2col_array(input.view(), kernel_size, stride_size, pad_size);
        println!("output = {:?}", output); //shape (1,4,9)
    }
```



## Im2col構造体
行列の処理ができたとして、Function構造体を実装していきます。Function構造体を実装するにあたり、バックプロパゲーションを考えなくてはならないのですが、実はこの **Im2col構造体** は前回のドキュメントの[Broaccast_to関数](https://chitono.github.io/StuCrs/book-basic/matrix_extension/broadcast_taiou/broadcast_to.html)が**sum_to関数** とバックプロパゲーションで表裏一体の関係だったのと同じように、今回ももう一つ別の構造体を同時に作ります。それが **Col2im** です。


```rust
struct Im2col {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
    generation: i32,
    id: usize,
}

impl Function for Im2col {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Im2colは一変数関数です。inputsの個数が一つではありません。")
        }

        let output = self.forward(inputs);

        if get_grad_status() == true {
            //inputのgenerationで一番大きい値をFuncitonのgenerationとする
            self.generation = inputs.iter().map(|input| input.generation()).max().unwrap();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる
            output.0.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: &[RcVariable]) -> RcVariable {
        let x = &xs[0];
        //ここは後に動的のままim2colに渡す予定。
        let x_data = x.data().into_dimensionality::<Ix4>().unwrap();

        let y_data = im2col_array(
            x_data.view(),
            self.kernel_size,
            self.stride_size,
            self.pad_size,
        );

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x_data = &self.inputs[0].data();
        let x_shape = x_data.shape();
        let x_shape = x_shape
            .try_into()
            .expect("Im2colのxの次元が4ではありません。");

        let gx = col2im_simple(
            gy,
            x_shape,
            self.kernel_size,
            self.stride_size,
            self.pad_size,
        );                 // broadcast_toのbackwardでsum_toを使うのと同じ
        let gxs = vec![gx];

        gxs
    }

    fn get_inputs(&self) -> &[RcVariable] {
        &self.inputs
    }

    fn get_output(&self) -> RcVariable {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        RcVariable(output)
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
}
impl Im2col {
    fn new(
        inputs: &[RcVariable],
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            kernel_size: kernel_size,
            stride_size: stride_size,
            pad_size: pad_size,
            generation: 0,
            id: id_generator(),
        }))
    }
}

fn im2col_f(
    xs: &[RcVariable],
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> RcVariable {
    Im2col::new(xs, kernel_size, stride_size, pad_size)
        .borrow_mut()
        .call()
}

pub fn im2col_simple(
    x: &RcVariable,
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> RcVariable {
    let y = im2col_f(&[x.clone()], kernel_size, stride_size, pad_size);
    y
}
```

Im2colのFunction構造体では、**Broadcast_to構造体** のバックプロパゲーションで **sum_to関数** を使ったのと同じようにバックプロパゲーションで **col2im関数** を用います。**col2im関数** は次のページで実装します。

ではIm2colのFunction構造体としての計算処理が正しいかテストします。特にバックプロパゲーションがうまく働くか確認します。
```rust
#[test]
    fn im2col_function_test() {
        use crate::core_new::ArrayDToRcVariable;

        let input = array![[[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]]]
        .rv();
        let kernel_size = (2, 2);
        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let mut output = im2col_simple(&input, kernel_size, stride_size, pad_size);

        println!("output = {:?}", output); //shape (1,4,9)

        output.backward(false);
        println!("input_grad = {:?}", input.grad().unwrap().data());
    }
```