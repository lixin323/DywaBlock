#!/usr/bin/env bash

set -ex

IMAGE_TAG='pkm1'
VERSION_TAG='v0'

# NOTE: Set context directory relative to this file.
CONTEXT_DIR="$( cd "$( dirname $(realpath "${BASH_SOURCE[0]}") )" && pwd )"

# 构建前准备 nvdiffrast 源码（避免镜像内从 GitHub clone 时的 TLS/网络问题）
if [ ! -d "${CONTEXT_DIR}/nvdiffrast/.git" ]; then
  echo "Cloning nvdiffrast into ${CONTEXT_DIR}/nvdiffrast ..."
  git clone --depth 1 https://github.com/NVlabs/nvdiffrast.git "${CONTEXT_DIR}/nvdiffrast"
fi

# 构建前准备 mvp 源码（同样避免镜像内访问 GitHub）
if [ ! -d "${CONTEXT_DIR}/mvp/.git" ]; then
  echo "Cloning mvp into ${CONTEXT_DIR}/mvp ..."
  git clone --depth 1 https://github.com/ir413/mvp.git "${CONTEXT_DIR}/mvp"
fi

# 构建前准备 pytorch-cosine-annealing-with-warmup 源码
if [ ! -d "${CONTEXT_DIR}/pytorch-cosine-annealing-with-warmup/.git" ]; then
  echo "Cloning pytorch-cosine-annealing-with-warmup into ${CONTEXT_DIR}/pytorch-cosine-annealing-with-warmup ..."
  git clone --depth 1 https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup.git "${CONTEXT_DIR}/pytorch-cosine-annealing-with-warmup"
fi

## Build docker image.
DOCKER_BUILDKIT=1 docker build --progress=plain \
    --network host \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    -t "${IMAGE_TAG}:${VERSION_TAG}" -f ${CONTEXT_DIR}/Dockerfile ${CONTEXT_DIR}
