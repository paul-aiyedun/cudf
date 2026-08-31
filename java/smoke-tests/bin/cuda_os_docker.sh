#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CUDA × OS Docker image map and build for run.sh. Source, do not execute.
#
# CUDA 12: 12.9.1-runtime on Ubuntu 22.04 / 24.04 / Rocky 8 (Ubuntu 26.04 unsupported).
# CUDA 13: 13.0.1-runtime on Ubuntu 22.04 / 24.04 / Rocky 8; 13.3.1 on Ubuntu 26.04.

base_image_for() {
  local cuda_major="$1"
  local os="$2"
  case "${cuda_major}:${os}" in
    12:ubuntu22.04)  echo "nvidia/cuda:12.9.1-runtime-ubuntu22.04" ;;
    12:ubuntu24.04)  echo "nvidia/cuda:12.9.1-runtime-ubuntu24.04" ;;
    12:rockylinux8)  echo "nvidia/cuda:12.9.1-runtime-rockylinux8" ;;
    12:ubuntu26.04)  return 1 ;;
    13:ubuntu22.04)  echo "nvidia/cuda:13.0.1-runtime-ubuntu22.04" ;;
    13:ubuntu24.04)  echo "nvidia/cuda:13.0.1-runtime-ubuntu24.04" ;;
    13:ubuntu26.04)  echo "nvidia/cuda:13.3.1-runtime-ubuntu26.04" ;;
    13:rockylinux8)  echo "nvidia/cuda:13.0.1-runtime-rockylinux8" ;;
    *)               return 1 ;;
  esac
}

dockerfile_for() {
  local os="$1"
  case "${os}" in
    ubuntu*)      echo "${SMOKE_TESTS_ROOT}/docker/Dockerfile.ubuntu" ;;
    rockylinux*)  echo "${SMOKE_TESTS_ROOT}/docker/Dockerfile.rocky" ;;
    *)            return 1 ;;
  esac
}

ensure_image() {
  local cuda_major="$1"
  local os="$2"
  local base_image dockerfile
  base_image="$(base_image_for "${cuda_major}" "${os}")"
  dockerfile="$(dockerfile_for "${os}")"
  local tag="cudf-java-smoke-test:cuda${cuda_major}-${os}"
  if ! docker image inspect "${tag}" >/dev/null 2>&1; then
    info "Building ${tag} (base=${base_image})"
    docker build \
      --build-arg "BASE_IMAGE=${base_image}" \
      -t "${tag}" \
      -f "${dockerfile}" \
      "${SMOKE_TESTS_ROOT}/docker" >&2
  fi
  echo "${tag}"
}

# Derive an image with every libnvrtc* moved out of the loader's search path, to
# exercise the JIT failure path on a host that lacks NVRTC. The files must be
# moved rather than renamed in place, otherwise ldconfig recreates the soname
# symlink and dlopen still succeeds.
ensure_image_without_nvrtc() {
  local base_tag="$1"
  local tag="${base_tag}-nonvrtc"
  if ! docker image inspect "${tag}" >/dev/null 2>&1; then
    info "Building ${tag} (NVRTC hidden)"
    docker build -t "${tag}" -f - "${SMOKE_TESTS_ROOT}/docker" >&2 << EOF
FROM ${base_tag}
USER root
RUN mkdir -p /opt/nvrtc-hidden && find / -xdev -name 'libnvrtc*' -print -exec mv -t /opt/nvrtc-hidden {} + && (ldconfig || true) && ! (ldconfig -p | grep -q nvrtc)
EOF
  fi
  echo "${tag}"
}
