#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
base="$PWD/.tools/opencv-stage-profile"
if [[ -f "$base/install/include/opencv5/opencv2/core.hpp" ]]; then exit 0; fi
mkdir -p "$base"
curl -L --fail https://codeload.github.com/opencv/opencv/tar.gz/refs/tags/5.0.0 -o "$base/source.tar.gz"
expected=b0528f5a1d379d59d4701cb28c36e22214cc51cf64594e5b56f2d3e6c0233095
actual=$(shasum -a 256 "$base/source.tar.gz" | cut -d ' ' -f 1)
[[ "$actual" == "$expected" ]] || { echo "OpenCV archive hash mismatch" >&2; exit 1; }
tar -xzf "$base/source.tar.gz" -C "$base"
cmake -S "$base/opencv-5.0.0" -B "$base/build" \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$base/install" \
 -DBUILD_LIST=core,imgproc -DBUILD_SHARED_LIBS=ON \
 -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
 -DBUILD_opencv_apps=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF \
 -DWITH_IPP=OFF -DWITH_OPENCL=OFF -DWITH_TBB=OFF \
 -DWITH_FFMPEG=OFF -DWITH_AVFOUNDATION=OFF -DWITH_ITT=OFF \
 -DWITH_LAPACK=OFF -DWITH_EIGEN=OFF -DOPENCV_GENERATE_PKGCONFIG=ON
cmake --build "$base/build" -j4
cmake --install "$base/build"
