#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Maven-repo, version, and classifier helpers for run.sh. Source, do not execute.
# Sets globals consumed by run.sh (ShellCheck does not follow sourced files).
# shellcheck disable=SC2034

discover_version() {
  local cudf_dir="${MAVEN_REPO}/ai/rapids/cudf"
  if [[ ! -d "${cudf_dir}" ]]; then
    die "No ai/rapids/cudf/ under ${MAVEN_REPO}"
  fi
  local -a versions=()
  local d
  for d in "${cudf_dir}"/*/; do
    if [[ ! -d "${d}" ]]; then
      continue
    fi
    versions+=("$(basename "${d}")")
  done
  if [[ "${#versions[@]}" -eq 0 ]]; then
    die "No version directories under ${cudf_dir}"
  fi
  if [[ "${#versions[@]}" -gt 1 ]]; then
    die "Multiple versions under ${cudf_dir}; pass --version. Found: ${versions[*]}"
  fi
  printf '%s\n' "${versions[0]}"
}

resolve_maven_repo() {
  if [[ "${MODE}" == "artifact-url" ]]; then
    MAVEN_REPO="$("${SCRIPT_DIR}/fetch_bundle.sh" --artifact-url "${ARTIFACT_URL}")"
  fi
  if [[ "${USE_CENTRAL}" -eq 0 ]]; then
    if [[ ! -d "${MAVEN_REPO}/ai/rapids/cudf" ]]; then
      die "No ai/rapids/cudf/ under ${MAVEN_REPO}"
    fi
    MAVEN_REPO="$(cd "${MAVEN_REPO}" && pwd)"
  fi
}

resolve_version() {
  if [[ "${USE_CENTRAL}" -eq 0 ]]; then
    if [[ -z "${VERSION}" ]]; then
      VERSION="$(discover_version)"
    elif [[ ! -d "${MAVEN_REPO}/ai/rapids/cudf/${VERSION}" ]]; then
      die "Version ${VERSION} not under ${MAVEN_REPO}/ai/rapids/cudf/"
    fi
  else
    if [[ -z "${VERSION}" ]]; then
      die "--use-maven-central requires --version"
    fi
  fi
}

classifiers_for_arch() {
  local cuda="${1:-}"
  case "$(uname -m)" in
    x86_64)
      case "${cuda}" in
        "")  echo "unclassified,cuda12,cuda13" ;;
        12)  echo "unclassified,cuda12" ;;
        13)  echo "cuda13" ;;
        *)   die "--cuda-version must be 12 or 13 (got: ${cuda})" ;;
      esac
      ;;
    aarch64|arm64)
      case "${cuda}" in
        "")  echo "cuda12-arm64,cuda13-arm64" ;;
        12)  echo "cuda12-arm64" ;;
        13)  echo "cuda13-arm64" ;;
        *)   die "--cuda-version must be 12 or 13 (got: ${cuda})" ;;
      esac
      ;;
    *) die "Unsupported arch $(uname -m)" ;;
  esac
}

jar_for_classifier() {
  local c="$1"
  if [[ "${c}" == "unclassified" ]]; then
    echo "${MAVEN_REPO}/ai/rapids/cudf/${VERSION}/cudf-${VERSION}.jar"
  else
    echo "${MAVEN_REPO}/ai/rapids/cudf/${VERSION}/cudf-${VERSION}-${c}.jar"
  fi
}

cuda_major_for_classifier() {
  case "$1" in
    *cuda13*) echo 13 ;;
    *)        echo 12 ;;
  esac
}
