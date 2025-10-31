# Dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    ca-certificates \
    libsqlite3-dev \
    clang \
    clang-tidy \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# libgtest-dev exists but needs building; we use FetchContent in CMake to pull gtest

WORKDIR /workspace
COPY . /workspace

# Build
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . -j$(nproc)

# Run tests
RUN cd build && ctest --output-on-failure

# Run a quick clang-tidy check on headers (optional)
RUN clang-tidy --version || true

CMD ["/bin/bash"]
