use crate::datasets::DatasetError;
use crate::functions_cnn::CNNError;
use crate::layers::LayerError;
use crate::optimizers::OptimizerError;
use crate::tensor::error::TensorError;
use std;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FrameError {
    #[error("行列計算(Tensor)における処理失敗")]
    Tensor(#[from] TensorError),

    #[error("Function構造体における{function}のforward()内での計算失敗")]
    ForwardError {
        function: &'static str,

        #[source]
        source: TensorError,
    },

    #[error("Function構造体における{function}のbackward()内での計算失敗")]
    BackwardError { function: &'static str },

    #[error("Function構造体における{function}は{expected}個の入力を期待しましたが、実際は{got}個渡されました")]
    InvalidInputCount {
        function: &'static str,

        expected: usize,
        got: usize,
    },

    #[error("CNNの計算における処理失敗")]
    CNNError(#[from] CNNError),

    #[error("Layer構造体の計算における処理失敗")]
    LayerError(#[from] LayerError),

    #[error("Optimizer構造体の計算における処理失敗")]
    OptimizerError(#[from] OptimizerError),

    #[error("Dataset構造体の計算における処理失敗")]
    DatasetError(#[from] DatasetError),

    #[error("{context}")]
    Unimplemented {
        context: String,
        #[source]
        source: Option<TensorError>,
    },
}

pub type FrameResult<T> = std::result::Result<T, FrameError>;
