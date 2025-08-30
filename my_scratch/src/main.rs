use ndarray::*;
use std::array;
use std::cell::RefCell;
use std::f32::consts::PI;
use std::rc::Rc;
use std::time::Instant;
use stucrs::core_new::{F32ToRcVariable, RcVariable};
use stucrs::datasets::*;
use stucrs::functions_new as F;
use stucrs::layers::{self as L, Activation, Dense, Layer, Linear};
use stucrs::models::{BaseModel, Model};
use stucrs::optimizers::{Optimizer, SGD};

use stucrs::core_new::ArrayDToRcVariable;

use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;

use plotters::prelude::*;
use rand::seq::SliceRandom;
use rand::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 3x4の2次元行列を作成
    // `mut`キーワードで可変にする
    let spiral = Spiral::new(true);

    let t = spiral.label;
    let x = spiral.data.view();

    for (i, batch) in x.axis_chunks_iter(Axis(0), 10).enumerate() {
        println!("i = {:?} batch = {:?}", i, batch);
    }

    let root = BitMapBackend::new("plot.png", (640, 640)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption("spiral", ("sans-serif", 40))
        .build_cartesian_2d(-1.0f32..1.0, -1.0f32..1.0)?;

    chart.configure_mesh().draw()?;
    chart.draw_series(
        spiral
            .data
            .rows()
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                if t[index] == 0 {
                    Circle::new((row[0], row[1]), 5, RED.filled())
                } else if t[index] == 1 {
                    Circle::new((row[0], row[1]), 5, BLUE.filled())
                } else {
                    Circle::new((row[0], row[1]), 5, GREEN.filled())
                }
            }),
    )?;

    Ok(())
}
