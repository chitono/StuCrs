use ndarray::array;
use std::fmt::Debug;
use thiserror::Error;

// ここはRustのふるまいを確認知る場所です。
fn main() {
    let a = array![[1.0, 2.0, 3.0]];
    let b = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let c = &a + &b;

    println!("{}", c);
}
fn divide(numerator: f64, denominator: f64) -> Result<f64, DivideByZero> {
    if denominator == 0.0 {
        Err(DivideByZero)
    } else {
        Ok(numerator / denominator)
    }
}

pub fn early_return() -> Result<(), String> {
    let value = divide(10.0, 0.0)?;
    println!("値は{}であり、中身が取り出されている", value);
    Ok(())
}

#[derive(Debug)]
struct DivideByZero;

impl From<DivideByZero> for String {
    fn from(_value: DivideByZero) -> Self {
        println!("convert DivideByZero to 'Divide by 0' String");
        "Divide By 0".to_string()
    }
}

impl std::error::Error for DivideByZero {}

impl std::fmt::Display for DivideByZero {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Divided By 0")
    }
}

#[derive(Debug)]
struct CustomErrorType1;

#[derive(Debug)]
struct CustomErrorType2;

#[derive(Debug)]
enum ApplicationError {
    Type1(CustomErrorType1),
    Type2(CustomErrorType2),
}

impl std::error::Error for ApplicationError {}

impl std::fmt::Display for ApplicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApplicationError::Type1(_) => write!(f, "Error type 1"),
            ApplicationError::Type2(_) => write!(f, "Error type 2"),
        }
    }
}

impl From<CustomErrorType1> for ApplicationError {
    fn from(value: CustomErrorType1) -> Self {
        ApplicationError::Type1(value)
    }
}

impl From<CustomErrorType2> for ApplicationError {
    fn from(value: CustomErrorType2) -> Self {
        ApplicationError::Type2(value)
    }
}

#[derive(Error, Debug)]
pub enum MyError {
    #[error("ファイルが見つかりません")]
    NotFound,

    #[error("無効な入力: {0}")]
    InvalidInput(String),
}

fn check(value: i32) -> Result<(), MyError> {
    if value < 0 {
        Err(MyError::InvalidInput("負の値".into()))
    } else {
        Ok(())
    }
}
