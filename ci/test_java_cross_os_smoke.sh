#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Cross-OS smoke test entrypoint for the packaged classifier JAR.
#
# Assembles the java-build artifact (single classifier subdir) into a
# Maven-repo layout, then runs the java/smoke-tests OS matrix via run.sh
# in Docker.
#
# When JAVA_PKG_DIR is unset, download the java-build artifact via
# rapids-download-from-github (pr.yaml for PRs, build.yaml otherwise).
#
# Inputs (environment variables):
#   JAVA_PKG_DIR   Optional. Path to a downloaded java-build artifact
#                  (one classifier subdir, e.g. java_pkg/cuda12/). If unset,
#                  the artifact is fetched via rapids-download-from-github.
#   CUDA_MAJOR     Optional. CUDA major version (12 or 13). If unset,
#                  derived from RAPIDS_CUDA_VERSION.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is required on the host runner (no job container)" >&2
  exit 1
fi

if [[ -z ${CUDA_MAJOR:-} ]]; then
  if [[ -z ${RAPIDS_CUDA_VERSION:-} ]]; then
    echo "Error: CUDA_MAJOR or RAPIDS_CUDA_VERSION must be set" >&2
    exit 1
  fi
  CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
fi

if [[ -z ${JAVA_PKG_DIR:-} ]]; then
  # matrix.ARCH values are amd64/arm64; $(arch) / uname -m return x86_64/aarch64.
  case "$(uname -m)" in
    x86_64)         java_arch=amd64 ;;
    aarch64|arm64)  java_arch=arm64 ;;
    *)
      echo "Error: unsupported host arch '$(uname -m)'" >&2
      exit 1
      ;;
  esac

  rapids-logger "Downloading cudf_java_${java_arch}_cu${CUDA_MAJOR}"
  JAVA_PKG_DIR="$(rapids-download-from-github "cudf_java_${java_arch}_cu${CUDA_MAJOR}")"
fi

if [[ ! -d ${JAVA_PKG_DIR} ]]; then
  echo "Error: JAVA_PKG_DIR='${JAVA_PKG_DIR}' is not a directory" >&2
  exit 1
fi

MAVEN_REPO="${RUNNER_TEMP:-/tmp}/maven-repo"
rm -rf "${MAVEN_REPO}"

echo "Assembling Maven repo from ${JAVA_PKG_DIR} -> ${MAVEN_REPO}"
"${REPO_ROOT}/java/ci/assemble_maven_repo.sh" \
  --jars-dir "${JAVA_PKG_DIR}" \
  --output-dir "${MAVEN_REPO}"

echo "Running cudf-java cross-OS smoke test (CUDA ${CUDA_MAJOR})"
"${REPO_ROOT}/java/smoke-tests/bin/run.sh" \
  --maven-repo-dir "${MAVEN_REPO}" \
  --cuda-version "${CUDA_MAJOR}"
