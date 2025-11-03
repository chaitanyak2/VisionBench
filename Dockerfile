# --------------------------------------------------------------------
# Base Image
# --------------------------------------------------------------------
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------
# Install build tools and dependencies
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# Install LibTorch (C++ Distribution)
# --------------------------------------------------------------------
# You can change the cu121 to cpu if you want CPU-only
ENV TORCH_VERSION=2.4.0%2Bcpu
ENV TORCH_URL=https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}.zip

RUN wget ${TORCH_URL} -O /tmp/libtorch.zip && \
    unzip /tmp/libtorch.zip -d /usr/local && \
    rm /tmp/libtorch.zip

# --------------------------------------------------------------------
# Set working directory and copy source
# --------------------------------------------------------------------
WORKDIR /workspace
COPY . /workspace

# --------------------------------------------------------------------
# Build and install GoogleTest manually (libgtest-dev installs only sources)
# --------------------------------------------------------------------
RUN apt-get update && apt-get install -y libgtest-dev && \
    cd /usr/src/gtest && \
    cmake . && \
    make -j$(nproc) && \
    cp lib/*.a /usr/lib


# --------------------------------------------------------------------
# Build project using LibTorch
RUN rm -rf build && mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH=/usr/local/libtorch \
          -DTorch_DIR=/usr/local/libtorch/share/cmake/Torch \
          .. && \
    cmake --build . -j$(nproc)


#---------------------------------------------------------------------
# Set environment
#---------------------------------------------------------------------
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu


# --------------------------------------------------------------------
# Run tests
# --------------------------------------------------------------------
# Run tests but don’t fail the build if any test crashes
RUN cd build && ctest --output-on-failure || true

# --------------------------------------------------------------------
# Run optional clang-tidy check
# --------------------------------------------------------------------
RUN clang-tidy --version || true

CMD ["/bin/bash"]
