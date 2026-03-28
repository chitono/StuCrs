# データセットの用意

ここではデータセットをモデルが扱えるように処理、管理する構造体を実装していきます。    
今までのデータセットと言えば、ニューラルネットワークの構築の際に用いた\\(y = sin(2\pi\cdot x) + b\\)があります。これは私たちがテストするために用意した簡易的なデータです。しかし、深層学習で学習させるデータセットは莫大なデータ数で、とても複雑なものです。これを毎回人の手でニューラルネットワークにうまく変換することは大変です。なので、それらを一元的に処理する構造体を実装していきます。   

こちらの構造体も今までと同様に、**Datasetトレイト** を実装し、様々な構造体に継承させるので、Datasetトレイトを実装した構造体を**Dataset構造体** と呼ぶことにします。

先ほどと同じように **core.rs** ファイルと同じ階層に **dataset.rs** ファイルを追加します。モジュールとして認識してもらうよう、 **lib.rs**、**mod.rs** に **dataset.rs** の名前を追加しておきます。


```rust
pub trait Dataset {
    fn len(&self) -> usize;
    fn data_setup(&mut self);
}
```
こちらが基本的なトレイトとなります。はじめは比較的簡易的なデータセットを扱うので、初歩的な機能のみを実装させます。**fn len()** はデータ数を返し、**fn data_setup** は実際にデータを初期化する処理を行います。

## スパイラルデータの用意

では次に学習させるスパイラルデータを例に挙げて実際に実装していきます。
>スパイラルデータは「ゼロから作るDeep Learning3　フレームワーク編」で使用されているスパイラルデータを参考にしています。

```rust
pub struct Spiral {
    pub data: Array2<f32>,
    pub label: Array1<u32>,
}

impl Dataset for Spiral {
    fn data_setup(&mut self) {}

    /*
    fn get_item(&self, index: i32) -> ArrayViewD<f32> {
        self.data.view().into_dyn()
    }
    */
    fn len(&self) -> usize {
        self.data.shape()[0]
    }
}

impl Spiral {
    pub fn new() -> Self {
        let data_label = get_spiral_data();
        let data = data_label.0;
        let label = data_label.1;
        let spiral = Self {
            data: data,
            label: label,
        };
        spiral
    }
}

fn get_spiral_data() -> (Array2<f32>, Array1<u32>) {
    let data_len = 100;
    let num_class = 3;
    let input_dim = 2;

    let data_size = data_len * num_class;

    let mut x = Array2::zeros((data_size, input_dim));

    let mut t = Array1::zeros(data_size);
    //let normal_dist = Normal::new(0.0f32, 1.0).unwrap();

    for j in 0..num_class {
        for i in 0..data_len {
            let rate = i as f32 / data_len as f32;
            let radius = 1.0 * rate as f32;

            let mut rng = rand::thread_rng();
            let normal: f32 = rng.sample(StandardNormal);

            let theta = j as f32 * 4.0 + 4.0 * rate as f32 + normal * 0.2;

            let ix = data_len * j + i;
            let mut x_row_view = x.row_mut(ix);

            let row_array = array![radius as f32 * theta.sin(), radius as f32 * theta.cos()];

            x_row_view.assign(&row_array);
            t[ix] = j as u32;
        }
    }
    //一つ2次元の行列と一次元の行列の対となる行が同じ位置に来るようにシャッフルして新しい二つの行列を返す
    double_matrix_shuffle_rows_immutable(x.view(), t.view())
}

pub fn double_matrix_shuffle_rows_immutable(
    arr1: ArrayView2<f32>,
    arr2: ArrayView1<u32>,
) -> (Array2<f32>, Array1<u32>) {
    if arr1.nrows() != arr2.len() {
        panic!("arr1とarr2の行列の行数が異なります")
    }
    // 行のインデックスを作成 (0, 1, 2, ...arr1.nrows())
    let mut indices: Vec<usize> = (0..arr1.nrows()).collect();

    // インデックスをシャッフル
    indices.shuffle(&mut thread_rng());
    
    let new_arr1 = arr1.select(Axis(0), &indices);
    let new_arr2 = arr2.select(Axis(0), &indices);

    (new_arr1.to_owned(), new_arr2.to_owned())
}
```


スパイラルデータを扱う構造体を **spiral** 構造体とします。今回のデータの生成は比較的簡単なので、data_setup()を使わず、**get_spiral_data()** で直接生成してフィールドに保持させます。このデータをrustのグラフを扱うライブラリである **plotters** で表示させると、渦巻きのような模様が現れます。そして、この三色の色に分類するというわけです。このグラフ表示は[plottersを使う](../Supplement/Plotters.md)で解説します。    

そして、生成されたデータを正解ラベルとともにシャッフルしていきます。というのも、学習させるデータをシャッフルしてモデルに渡すことで、順番による影響を最小限にすることができます。ここで、データとラベルがずれることがないようにシャッフルする必要があります。この処理は **double_matrix_shuffle_rows_immutable()** で行います。今回は二次元の座標データなので、次元は2次元\\((x,y)\\)です。

<br>

$$

data:\begin{pmatrix}
x_0 & y_0  \\\\ 
x_1 & y_1  \\\\ 
x_2 & y_2  \\\\
x_3 & y_3  \\\\
x_4 & y_4  \\\\
x_5 & y_5  \\\\
\vdots & \vdots  \\\\
x_N & y_N
\end{pmatrix}
\qquad\
T:\begin{pmatrix}
t_0 \\\\ 
t_1 \\\\ 
t_2\\\\
t_3\\\\
t_4\\\\
t_5\\\\
\vdots\\\\
t_N
\end{pmatrix} 
\quad
\xrightarrow{\text{shuffle}}
\quad
data:\begin{pmatrix}
x_4 & y_4  \\\\ 
x_N & y_N  \\\\ 
x_2 & y_2  \\\\
x_3 & y_3  \\\\
x_1 & y_1  \\\\
x_0 & y_0  \\\\
\vdots & \vdots  \\\\
x_5 & y_5
\end{pmatrix}
\qquad\
T:\begin{pmatrix}
t_4 \\\\ 
t_N \\\\ 
t_2\\\\
t_3\\\\
t_1\\\\
t_0\\\\
\vdots\\\\
t_5
\end{pmatrix} 
$$

<br>    

これはあくまでシャッフルの一例ですが、大事なのは、データとラベルの順番がしっかり対応しているということです。これにより、シャッフルされた状態でも、同じデータとラベルの組を正しく取り出すことができます。   

この学習データは次の学習で用いるので、実装しておきましょう。