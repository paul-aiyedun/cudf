#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Smoke test cudf-java classifier JARs (cudf + slf4j classpath) in Docker,
# across the RAPIDS OS matrix (Ubuntu 22.04/24.04/26.04 + Rocky Linux 8).
#
# Exactly one source mode:
#   --artifact-url URL | --maven-repo-dir DIR | --use-maven-home | --use-maven-central

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_TESTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${SMOKE_TESTS_ROOT}/.cache"

# shellcheck source=logging.sh
. "${SCRIPT_DIR}/logging.sh"
# shellcheck source=run_args.sh
. "${SCRIPT_DIR}/run_args.sh"
# shellcheck source=cudf_maven.sh
. "${SCRIPT_DIR}/cudf_maven.sh"
# shellcheck source=cuda_os_docker.sh
. "${SCRIPT_DIR}/cuda_os_docker.sh"

DEFAULT_OS_LIST="ubuntu22.04,ubuntu24.04,ubuntu26.04,rockylinux8"
OS_LIST="${DEFAULT_OS_LIST}"

MODE=""
MAVEN_REPO=""
VERSION=""
ARTIFACT_URL=""
CUDA_VERSION=""
USE_CENTRAL=0

CLASSIFIERS=""
MVN_ARGS=()

set_mvn_args() {
  local classifier="$1"
  MVN_ARGS=(mvn -B "-Dcudf.version=${VERSION}")
  if [[ "${classifier}" == "unclassified" ]]; then
    MVN_ARGS+=(-Dcudf.unclassified=true)
  else
    MVN_ARGS+=("-Dcuda.classifier=${classifier}")
  fi
  MVN_ARGS+=(package exec:java)
}

docker_run_smoke_test() {
  local tag="$1"
  local log_file="$2"

  local -a docker_cmd=(
    docker run --rm --gpus all
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp
    -e MAVEN_OPTS=-Duser.home=/tmp
    -v "${SMOKE_TESTS_ROOT}:/smoke-test"
    -v "${CACHE_DIR}/m2-container:/tmp/.m2"
    -w /smoke-test
  )
  local -a mvn_extra=(-Dmaven.repo.local=/tmp/.m2/repository)
  if [[ "${USE_CENTRAL}" -eq 1 ]]; then
    mvn_extra+=(-Dcudf.smoke-test.central=true)
  else
    docker_cmd+=(-v "${MAVEN_REPO}:/maven-repo:ro")
    mvn_extra+=(-Dcudf.maven.repo=/maven-repo)
  fi

  "${docker_cmd[@]}" "${tag}" "${MVN_ARGS[@]}" "${mvn_extra[@]}" >"${log_file}" 2>&1
}

run_smoke_test() {
  local classifier="$1"
  local os="$2"
  local log_file="${CACHE_DIR}/logs/smoke-test-${classifier}-${os}.log"
  mkdir -p "${CACHE_DIR}/logs" "${CACHE_DIR}/m2-container"

  local cuda_major tag
  cuda_major="$(cuda_major_for_classifier "${classifier}")"
  tag="$(ensure_image "${cuda_major}" "${os}")"
  set_mvn_args "${classifier}"

  info "Smoke test classifier=${classifier} os=${os} version=${VERSION}"
  set +e
  docker_run_smoke_test "${tag}" "${log_file}"
  local rc=$?
  set -e
  cat "${log_file}"
  return "${rc}"
}

run_matrix() {
  local -a classifier_list os_list
  IFS=',' read -r -a classifier_list <<< "${CLASSIFIERS}"
  IFS=',' read -r -a os_list <<< "${OS_LIST}"

  local failed=0 ran=0
  local classifier os jar cuda_major
  for classifier in "${classifier_list[@]}"; do
    if [[ "${USE_CENTRAL}" -eq 0 ]]; then
      jar="$(jar_for_classifier "${classifier}")"
      if [[ ! -f "${jar}" ]]; then
        warn "Skipping missing classifier '${classifier}' (${jar})"
        continue
      fi
    fi
    cuda_major="$(cuda_major_for_classifier "${classifier}")"
    for os in "${os_list[@]}"; do
      if ! base_image_for "${cuda_major}" "${os}" >/dev/null; then
        warn "Skipping unsupported combination: classifier=${classifier} (cuda${cuda_major}) os=${os}"
        continue
      fi
      ran=$((ran + 1))
      if run_smoke_test "${classifier}" "${os}"; then
        info "PASS: ${classifier} on ${os}"
      else
        warn "FAIL: ${classifier} on ${os}"
        failed=1
      fi
    done
  done

  if [[ "${ran}" -eq 0 ]]; then
    die "No classifier/OS combinations received a smoke test"
  fi
  if [[ "${failed}" -ne 0 ]]; then
    die "One or more classifier smoke tests failed"
  fi
  info "All classifier smoke tests passed"
}

main() {
  parse_args "$@"

  if ! command -v docker >/dev/null 2>&1; then
    die "docker is required"
  fi

  resolve_maven_repo
  resolve_version
  CLASSIFIERS="$(classifiers_for_arch "${CUDA_VERSION}")"

  info "Mode:        ${MODE}"
  if [[ "${MODE}" == "artifact-url" ]]; then
    info "Artifact:    ${ARTIFACT_URL}"
  fi
  if [[ "${USE_CENTRAL}" -eq 0 ]]; then
    info "Maven repo:  ${MAVEN_REPO}"
  fi
  info "Version:     ${VERSION}"
  info "Classifiers: ${CLASSIFIERS}"
  info "OSes:        ${OS_LIST}"

  # Force Maven to resolve cudf from the source currently under test.
  rm -rf "${CACHE_DIR}/m2-container/repository/ai/rapids/cudf/${VERSION:?}"

  run_matrix
}

main "$@"
