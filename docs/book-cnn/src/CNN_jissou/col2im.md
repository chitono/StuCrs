# Col2im関数の実装

では **Im2col** のバックプロパゲーションとなる関数の **Col2im関数** を実装していきます。基本的な流れは **Im2col** の時の同じです。




それではまず行列自体における計算についてです。

## Col2imの行列計算

Col2imという関数の名前は、フィルターで抽出できるような特殊な行列 **col** を入力データである **Image** に変換する関数というのが由来となっています。まさに先ほどの **Im2col** とは真逆の存在です。

Col2imの行列の処理はim2colでフィルターによって取り出された回数ととらえることができます。あるインデックスのデータがフィルターの移動によって何回読み込まれたか、もっと言えば、生成された **col** の行列にそのインデックス由来のデータがいくつ存在するかということです。**sum_to関数** の拡張したぶん、バックプロパゲーションで足し合わせる計算と近しいです。では実際に処理するコードを表示します。   

```rust
/// colsからimageに変更する関数。
/// inputにはcolsを渡す。
/// im_shapeは元のimageのshape(N,C,H,W)を渡す。
pub fn col2im_array(
    input: ArrayView3<f32>,
    im_shape: [usize; 4],
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> Array4<f32> {
    let (kh, kw) = kernel_size;
    let (stride_h, stride_w) = stride_size;
    let (pad_h, pad_w) = pad_size;

    let (n, c, h, w) = (im_shape[0], im_shape[1], im_shape[2], im_shape[3]); //元のimageの形状を取得。
    let (oh, ow) = get_conv_outsize((h, w), (kh, kw), (stride_h, stride_w), (pad_h, pad_w));

    let mut imgs = Array4::<f32>::zeros((n, c, h, w));

    for b in 0..n {
        let col = input.slice(s![b, .., ..]);
        let mut img = imgs.slice_mut(s![b, .., .., ..]);
        let mut col_idx = 0;

        for y in 0..oh {
            for x in 0..ow {
                let y_start = y as isize * stride_h as isize - pad_h as isize;
                let x_start = x as isize * stride_w as isize - pad_w as isize;

                let mut patch_row_idx = 0;

                for c_idx in 0..c {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let in_y = y_start + ky as isize;
                            let in_x = x_start + kx as isize;

                            // paddingしていないところか判定。
                            if in_y >= 0 && (in_y as usize) < h && in_x >= 0 && (in_x as usize) < w
                            {
                                let value = col[(patch_row_idx, col_idx)];
                                // imgの対応するところに加算する。
                                img[(c_idx, in_y as usize, in_x as usize)] += value; // 足し合わせる
                            }
                            patch_row_idx += 1;
                        }
                    }
                }

                col_idx += 1;
            }
        }
    }
    imgs
}
```



**Im2col** のところで説明した入力データを3次元の行列に変換した **col** ではなく、**Col2im** では 入力画像である **imgs**を初期化します。この時入力画像は4次元で、行列の形状が入力データと同じだとわかります。基本的な処理は先ほどと同じですが、 **value** で微分のデータを取り出し、足し合わせています。

では先ほどと同じように計算の確認をします。
```rust
#[test]
    fn col2im_test() {
        use crate::functions_cnn::col2im_array;

        // im2col_testの出力。(output)
        let input = array![[
            [1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0],
            [2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0],
            [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0],
            [6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]
        ]];

        let kernel_size = (2, 2);
        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let output = col2im_array(
            input.view(),
            [1, 1, 4, 4],
            kernel_size,
            stride_size,
            pad_size,
        );
        println!("output = {:?}", output);
        /*output = [[[[1.0, 4.0, 6.0, 4.0],
        [10.0, 24.0, 28.0, 16.0],
        [18.0, 40.0, 44.0, 24.0],
        [13.0, 28.0, 30.0, 16.0]]]] */
    }
```



## Col2im構造体
行列の処理ができたので、Function構造体を実装していきます。

```rust
struct Col2im {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    input_shape: [usize; 4],
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
    generation: i32,
    id: usize,
}

impl Function for Col2im {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Col2imは一変数関数です。inputsの個数が一つではありません。")
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
        //ここは後に動的のままcol2imに渡す予定。
        let x_data = x.data().into_dimensionality::<Ix3>().unwrap();

        let y_data = col2im_array(
            x_data.view(),
            self.input_shape,
            self.kernel_size,
            self.stride_size,
            self.pad_size,
        );

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let gx = im2col_simple(gy, self.kernel_size, self.stride_size, self.pad_size);
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
impl Col2im {
    fn new(
        inputs: &[RcVariable],
        input_shape: [usize; 4],
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            input_shape: input_shape,
            kernel_size: kernel_size,
            stride_size: stride_size,
            pad_size: pad_size,
            generation: 0,
            id: id_generator(),
        }))
    }
}

fn col2im_f(
    xs: &[RcVariable],
    input_shape: [usize; 4],
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> RcVariable {
    Col2im::new(xs, input_shape, kernel_size, stride_size, pad_size)
        .borrow_mut()
        .call()
}

pub fn col2im_simple(
    x: &RcVariable,
    input_shape: [usize; 4],
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> RcVariable {
    let y = col2im_f(
        &[x.clone()],
        input_shape,
        kernel_size,
        stride_size,
        pad_size,
    );
    y
}
```

**Im2col関数** と **col2im関数** の両方が実装できたので、二つともバックプロパゲーションのテストを行えます。


ではCol2imのFunction構造体の計算処理が正しいかテストします。特にバックプロパゲーションがうまく働くか確認します。
```rust
#[test]
    fn col2im_function_test() {
        use crate::core_new::ArrayDToRcVariable;

        // im2col_testの出力。(output)
        let input = array![[
            [1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0],
            [2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0],
            [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0],
            [6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]
        ]]
        .rv();

        let kernel_size = (2, 2);
        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let input_shape = [1, 1, 4, 4];

        let mut output = col2im_simple(&input, input_shape, kernel_size, stride_size, pad_size);

        println!("output = {:?}", output);
        /*output = [[[[1.0, 4.0, 6.0, 4.0],
        [10.0, 24.0, 28.0, 16.0],
        [18.0, 40.0, 44.0, 24.0],
        [13.0, 28.0, 30.0, 16.0]]]] */

        output.backward(false);
        println!("input_grad = {:?}", input.grad().unwrap().data());
    }
```


畳み込み処理の中核を担う関数を実装できたので、次は畳み込みを実際に行う **Conv2d構造体** を実装していきます。