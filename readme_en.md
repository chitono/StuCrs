# StuCrs: A Deep Learning Framework fully implemented from Scratch in Rust

>*This README is available in both Japanese and English to help bridge the gap between our local community and the global research landscape.*   
[日本語](README.md) / [English](readme_en.md)


## Overview

In this research, we developed "StuCrs," a deep learning framework implemented from scratch using the Rust programming language. StuCrs is characterized by its pure-Rust implementation and a simple structure designed to develop an intuitive understanding of the underlying principles.
It serves as an educational resource designed for "learning through reimplementation and new perspectives" encouraging users to build deep learning systems from the ground up to deepen their understanding of its mechanics. It also serves as an excellent sample codebase for those looking to learn Rust.




## Philosophy


*This project does not treat localization as a limitation,
but as a source of new ideas and perspectives.*


- **Learning Through Reimplementation**   
  This project encourages users not just to use the framework, but to rebuild it from scratch. By reconstructing each component, learners can actively explore and internalize the fundamental principles of deep learning.
  

<br>

- **Bridging the Language Gap in Technology**:    
  Most major frameworks like TensorFlow or PyTorch rely heavily on English-centric documentation and communities. By providing high-quality Japanese explanations and a framework built with local developers in mind, this project aims to lower the barrier to entry and enable deeper conceptual understanding of deep learning.    


<br>

- **Pioneering Deep Learning in Rust:**    
  While the machine learning ecosystem in Python and C++ is already mature, Rust introduces fundamentally different design principles. This project aims to stimulate the growth of local ML communities in Rust and promote Rust-based development in Japan and beyond.   


<br> 

- **A Foundation for Theoretical Exploration:**    
  StuCrs is not only an implementation, but also a platform for exploring new theoretical perspectives in deep learning. Based on Rust’s design philosophy, this project explores whether different abstraction models and system constraints can lead to new theoretical insights.  

<br> 

>**Research Vision:**       
>My goal is to foster a strong local research and developer ecosystem in Japan while contributing to the global Rust and ML communities. I believe that strengthening local research ecosystems is the first step toward meaningful global collaboration.


## Background

- Most major frameworks (e.g., TensorFlow, PyTorch) are English-centric, which can be a barrier for non-English-speaking learners.

- The Rust ecosystem for machine learning is still less mature compared to Python or C++.

- This project explores whether implementing deep learning in a modern systems language can offer new perspectives.


## Acknowledgements

<div style="display:flex; gap:100px;">
  <img style="width:31%;" alt="Image" src="https://github.com/user-attachments/assets/4bd79c56-4805-471d-af1c-af0c2fb5d890" />
  <img style="width:15%;" alt="Image" src="https://github.com/user-attachments/assets/d5d1ca74-79cb-4de3-b55c-537c705788f7" />
</div>   

<br>

This implementation is based on the book:  
`"ゼロから作るDeep Learning ③ - フレームワーク編"` by Koki Saitoh   


You can find the official repository here:[oreilly-japan/deep-learning-from-scratch-3](https://github.com/oreilly-japan/deep-learning-from-scratch-3)

We sincerely appreciate the author for granting permission to reference and adapt ideas from the book. The original framework `Dezero` was also used as a reference.

## Getting Started

### Dependencies (External Crates)
- [ndarray-0.16.0](https://docs.rs/ndarray/0.16.0/ndarray/index.html)
- [ndarray_stats-0.6.0](https://docs.rs/ndarray-stats/0.6.0/ndarray_stats/index.html)
- [ndarray-rand-0.15.0](https://docs.rs/ndarray-rand/latest/ndarray_rand/index.html)
- [mnist-0.6.0](https://docs.rs/mnist/latest/mnist/index.html)
- [rand-0.8](https://docs.rs/rand/latest/rand/index.html)
- [rand_distr-0.4.0](https://docs.rs/rand_distr/0.4.0/rand_distr/index.html)
- [fxhash-0.2.1](https://docs.rs/fxhash/latest/fxhash/index.html)

### Running with Docker
We recommend using Docker for a consistent environment.

1. Launch the container using the provided `Dockerfile` and `compose.yaml`.
2. Import the `stucrs` folder as an external crate in your project.


### GPU Acceleration (Work in Progress)   
The GPU support module is currently under active development. We are looking for contributors to help with GPU support. please join our [Discussions](https://github.com/chitono/StuCrs/discussions))

Please refer to the [Basic Documentation](https://chitono.github.io/StuCrs/book-basic/) for details.





## Documentation (Japanese)

Detailed explanations of the implementation process are available here.
- [『StuCrs Book・Basic』](https://chitono.github.io/StuCrs/book-basic/)
- [『StuCrs Book・CNN』](https://chitono.github.io/StuCrs/book-cnn/)




## Code Example: Training MNIST
Please refer to the [examples](examples) for other codes.
```

use ndarray::*;

use rand::seq::SliceRandom;
use rand::*;

use std::time::Instant;
use stucrs::config;
use stucrs::core_new::ArrayDToRcVariable;

use stucrs::datasets::*;
use stucrs::functions_new::{self as F, accuracy};
use stucrs::layers::{self as L, Activation};
use stucrs::models::{BaseModel, Model};
use stucrs::optimizers::{Optimizer, SGD};

fn main() {
    let mnist = MNIST::new();
    let x_train = mnist.train_img.view();
    let y_train = mnist.train_label.view();
    let x_test = mnist.test_img.view();
    let y_test = mnist.test_label.view();

    

    let x_train = x_train.to_shape((50000, 28 * 28)).unwrap();
    let x_test = x_test.to_shape((10000, 28 * 28)).unwrap();

    let y_train = arr2d_to_one_hot(y_train.mapv(|x| x as u32).view(), 10);
    let y_test = arr2d_to_one_hot(y_test.mapv(|x| x as u32).view(), 10);

    let max_epoch = 5;
    let lr = 0.01;
    let batch_size = 100;

    let data_size = x_train.shape()[0];
    println!("data_size={}", data_size);

    let mut model = BaseModel::new();
    model.stack(L::Dense::new(1000, true, None, Activation::Relu));
    model.stack(L::Dense::new(1000, true, None, Activation::Relu));
    model.stack(L::Linear::new(10, true, None));

    let mut optimizer = SGD::new(lr);
    optimizer.setup(&model);
    let start = Instant::now();
    for epoch in 0..max_epoch {

        .
        .
        .

```


## Performance

Performance benchmarks are available in
[Performance](PERFORMANCE.md).


## Final Notes  
This project was initiated as an independent research by high school students and it is still evolving.

As such, there may be:
- non-idiomatic Rust code
- inefficiencies
- misunderstandings in deep learning theory

Feedback is highly appreciated.


## Feedback
We welcome all kinds of feedback:
- Your background (student, researcher, engineer, etc.)
- Interesting aspects
- Confusing parts
- Suggestions or related work

Please use Discussions: [Discussions](https://github.com/chitono/StuCrs/discussions)
