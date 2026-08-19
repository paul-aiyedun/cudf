#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Cross-OS smoke test entrypoint for the packaged classifier JAR.
#
# Assembles the java-build artifact (single classifier subdir) into a
# Maven-repo layout, then runs the java/smoke-tests OS matrix via run.sh
# in Docker.
#
# Inputs (environment variables):
#   JAVA_PKG_DIR      Path to the downloaded java-build artifact (required).
#                     Contains one classifier subdir, e.g. java_pkg/cuda12/.
#   CUDA_MAJOR        CUDA major version (12 or 13). Narrows classifiers
#                     tried by run.sh to match this cell.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z ${JAVA_PKG_DIR:-} || ! -d ${JAVA_PKG_DIR} ]]; then
  echo "Error: JAVA_PKG_DIR must point to an existing java-build artifact dir" >&2
  exit 1
fi

if [[ -z ${CUDA_MAJOR:-} ]]; then
  echo "Error: CUDA_MAJOR must be set (12 or 13)" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is required on the host runner (no job container)" >&2
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
