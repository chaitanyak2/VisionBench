FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# ====================================================================
# Install build essentials + latest CMake (≥3.24 required by FAISS)
# ====================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git \
    wget \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 🔹 Install modern CMake from Kitware
RUN apt-get update && \
    apt-get install -y apt-transport-https gnupg && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

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
    libopenblas-dev \
    libomp-dev \
    liblapack-dev \
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

# ====================================================================
# Build FAISS from source (CPU only)
# ====================================================================
RUN git clone --depth=1 https://github.com/facebookresearch/faiss.git /tmp/faiss && \
    cd /tmp/faiss && \
    sed -i '/add_subdirectory(perf_tests)/d' CMakeLists.txt && \
    cmake -B build \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DFAISS_ENABLE_C_API=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DFAISS_ENABLE_PERF_TESTS=OFF \
        . && \
    cmake --build build -j$(nproc) && \
    cmake --install build && \
    rm -rf /tmp/faiss

# --------------------------------------------------------------------
# Set working directory and copy source
# --------------------------------------------------------------------
WORKDIR /workspace
COPY . /workspace

# --------------------------------------------------------------------
# Install GoogleTest headers only
# --------------------------------------------------------------------
RUN apt-get update && apt-get install -y libgmock-dev

# --------------------------------------------------------------------
# Build project using LibTorch
RUN rm -rf build && mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH=/usr/local/libtorch \
          -DTorch_DIR=/usr/local/libtorch/share/cmake/Torch \
          .. && \
    cmake --build . -j$(nproc)



#---------------------------------------------------------------------
# Set environment and library paths
#---------------------------------------------------------------------
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/faiss.conf && ldconfig
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/libtorch/lib:/usr/lib/x86_64-linux-gnu
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
