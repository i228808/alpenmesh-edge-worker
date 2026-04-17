# Stage 1: Builder
FROM rust:1.80-bullseye as builder
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y cmake pkg-config libssl-dev build-essential

# Copy source code
COPY . .

# Build the release binary
RUN cargo build --release --bin alpenmesh-worker

# Stage 2: Runtime
# Uses official NVIDIA image with CUDA, cuDNN, and TensorRT pre-installed
FROM nvcr.io/nvidia/tensorrt:23.08-py3
WORKDIR /app

# Install runtime dependencies (ffmpeg for stream decoding, certificates for HTTPS, libssl for reqwest)
RUN apt-get update && apt-get install -y ffmpeg ca-certificates libssl-dev && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary from builder
COPY --from=builder /build/target/release/alpenmesh-worker /app/alpenmesh-worker

# Copy the ONNX model and other assets
COPY ./assets /app/assets
COPY ./app.yaml /app/app.yaml

# Create directory for identity storage
RUN mkdir -p /app/data

# Configure TensorRT and CUDA paths for ONNX Runtime
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Default configuration pointing back to the host machine for testing
# host.docker.internal routes back to the host's localhost (Scheduler/Economy)
ENV SCHEDULER_URL=http://host.docker.internal:3000
ENV ECONOMY_URL=http://host.docker.internal:3010
ENV WORKER_EXTERNAL_ADDR=127.0.0.1:8081
ENV WORKER_IDENTITY_PATH=/app/data/worker_identity.json
ENV RUST_LOG=info,alpenmesh_worker=debug

# Expose the worker port
EXPOSE 8081

# Start the worker
ENTRYPOINT ["/app/alpenmesh-worker"]