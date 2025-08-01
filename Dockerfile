FROM  nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH="/root/.cargo/bin:${PATH}"

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update &&  apt-get install -y --no-install-recommends \
    tzdata \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    zlib1g-dev \
    libtinfo-dev \
    libxml2-dev \
    clang \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && rustup update stable

RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/cuda/compat" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/cuda/nvvm/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    ldconfig

RUN rustup component add rust-src
RUN rustup component add rust-analysis
RUN rustup toolchain install stable
RUN rustup default stable
RUN rustup component add rustfmt

WORKDIR /workdir


