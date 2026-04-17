# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV CARGO_NET_RETRY=10
ENV CARGO_HTTP_MULTIPLEXING=false
ENV CARGO_HTTP_TIMEOUT=600
ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    pkg-config \
    cmake \
    clang \
    lld \
    git \
    ffmpeg \
    libssl-dev \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    libnvonnxparsers-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p "${CARGO_HOME}" "${RUSTUP_HOME}" && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal --default-toolchain stable

RUN which cargo && cargo --version && rustc --version

WORKDIR /build
COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo build --release --bin alpenmesh-worker && \
    strip target/release/alpenmesh-worker

FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libssl3t64 \
    libnvinfer10 \
    libnvinfer-plugin10 \
    libnvonnxparsers10 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/alpenmesh-worker /app/alpenmesh-worker
COPY ./assets /app/assets
COPY ./app.yaml /app/app.yaml

RUN mkdir -p /app/data

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
ENV SCHEDULER_URL=http://host.docker.internal:3000
ENV ECONOMY_URL=http://host.docker.internal:3010
# Set WORKER_EXTERNAL_ADDR at runtime when the scheduler runs on another machine (see README_v2.md).
ENV WORKER_IDENTITY_PATH=/app/data/worker_identity.json
ENV RUST_LOG=info,alpenmesh_worker=debug

EXPOSE 8081
ENTRYPOINT ["/app/alpenmesh-worker"]