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

DEFAULT_OS_LIST="ubuntu22.04,ubuntu24.04,ubuntu26.04,rockylinux8"

MODE=""
MAVEN_REPO=""
VERSION=""
ARTIFACT_URL=""
CUDA_VERSION=""
OS_LIST=""
USE_CENTRAL=0

# Populated by resolvers / run_matrix.
CLASSIFIERS=""
MVN_ARGS=()

info() { printf '==> %s\n' "$*" >&2; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

require_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "${value}" ]]; then
    die "${flag} requires a value"
  fi
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "${s}"
}

# split_csv OUT_ARRAY_NAME CSV
split_csv() {
  local -n _out_arr="$1"
  local csv="$2"
  local -a parts
  IFS=',' read -r -a parts <<< "${csv}"
  _out_arr=()
  local p trimmed
  for p in "${parts[@]}"; do
    trimmed="$(trim "${p}")"
    if [[ -n "${trimmed}" ]]; then
      _out_arr+=("${trimmed}")
    fi
  done
}

set_mode() {
  local next="$1"
  if [[ -n "${MODE}" ]]; then
    die "Conflicting source modes: already using --${MODE}, cannot also use --${next}"
  fi
  MODE="${next}"
}

print_help() {
  cat << EOF
Usage: $(basename "$0") --artifact-url URL [options]
       $(basename "$0") --maven-repo-dir DIR [options]
       $(basename "$0") --use-maven-home [options]
       $(basename "$0") --use-maven-central --version VER [options]

Runs smoke tests in Docker across the RAPIDS OS matrix (Ubuntu 22.04/24.04/26.04
+ Rocky Linux 8). CUDA 12 x Ubuntu 26.04 is skipped (NVIDIA does not publish
that base image; RAPIDS does not test that pair).

Source modes (exactly one required):
  --artifact-url URL      Fetch gather artifact via fetch_bundle.sh, then test
  --maven-repo-dir DIR    Local Maven tree (must contain ai/rapids/cudf/)
  --use-maven-home        Use ~/.m2/repository
  --use-maven-central     Resolve from Maven Central (--version required)

Options:
  --version VER           ai.rapids:cudf version
                          (required for --use-maven-central; otherwise if
                          omitted, exactly one version must exist under
                          ai/rapids/cudf/)
  --cuda-version 12|13    Narrow classifiers to this CUDA major
  --os LIST               Comma-separated OS list
                          (default: ${DEFAULT_OS_LIST})
  -h, --help
EOF
}

# ========= arg parsing =========

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) print_help; exit 0 ;;
      --artifact-url)
        require_value "$1" "${2:-}"
        set_mode "artifact-url"
        ARTIFACT_URL="$2"
        shift 2
        ;;
      --maven-repo-dir)
        require_value "$1" "${2:-}"
        set_mode "maven-repo-dir"
        MAVEN_REPO="$2"
        shift 2
        ;;
      --use-maven-home)
        set_mode "use-maven-home"
        MAVEN_REPO="${HOME}/.m2/repository"
        shift
        ;;
      --use-maven-central)
        set_mode "use-maven-central"
        USE_CENTRAL=1
        shift
        ;;
      --version)
        require_value "$1" "${2:-}"
        VERSION="$2"
        shift 2
        ;;
      --cuda-version)
        require_value "$1" "${2:-}"
        CUDA_VERSION="$2"
        shift 2
        ;;
      --os)
        require_value "$1" "${2:-}"
        OS_LIST="$2"
        shift 2
        ;;
      *) die "Unknown argument: $1 (try --help)" ;;
    esac
  done

  if [[ -z "${MODE}" ]]; then
    die "Exactly one source mode is required (try --help)"
  fi
  if [[ -z "${OS_LIST}" ]]; then
    OS_LIST="${DEFAULT_OS_LIST}"
  fi
}

# ========= maven repo / version resolution =========

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

# ========= classifier + arch selection =========

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

# ========= OS × CUDA base-image map =========
#
# CUDA 12: 12.9.1-runtime on Ubuntu 22.04 / 24.04 / Rocky 8 (Ubuntu 26.04 unsupported).
# CUDA 13: 13.0.1-runtime on Ubuntu 22.04 / 24.04 / Rocky 8; 13.3.1 on Ubuntu 26.04
#          (only CUDA 13.3.x publishes an ubuntu26.04 base image).

base_image_for() {
  local major="$1"
  local os="$2"
  case "${major}:${os}" in
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
  local major="$1"
  local os="$2"
  local base_image dockerfile
  base_image="$(base_image_for "${major}" "${os}")"
  dockerfile="$(dockerfile_for "${os}")"
  local tag="cudf-java-smoke:cuda${major}-${os}"
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

# ========= mvn + docker invocation =========

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

docker_run_smoke() {
  local tag="$1"
  local log_file="$2"

  local -a docker_cmd=(
    docker run --rm --gpus all
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp
    -e MAVEN_OPTS=-Duser.home=/tmp
    -v "${SMOKE_TESTS_ROOT}:/smoke"
    -v "${CACHE_DIR}/m2-container:/tmp/.m2"
    -w /smoke
  )
  local -a mvn_extra=(-Dmaven.repo.local=/tmp/.m2/repository)
  if [[ "${USE_CENTRAL}" -eq 1 ]]; then
    mvn_extra+=(-Dcudf.smoke.central=true)
  else
    docker_cmd+=(-v "${MAVEN_REPO}:/maven-repo:ro")
    mvn_extra+=(-Dcudf.maven.repo=/maven-repo)
  fi

  "${docker_cmd[@]}" "${tag}" "${MVN_ARGS[@]}" "${mvn_extra[@]}" >"${log_file}" 2>&1
}

run_smoke() {
  local classifier="$1"
  local os="$2"
  local log_file="${CACHE_DIR}/logs/smoke-${classifier}-${os}.log"
  mkdir -p "${CACHE_DIR}/logs" "${CACHE_DIR}/m2-container"

  local major tag
  major="$(cuda_major_for_classifier "${classifier}")"
  tag="$(ensure_image "${major}" "${os}")"
  set_mvn_args "${classifier}"

  info "Smoke test classifier=${classifier} os=${os} version=${VERSION}"
  set +e
  docker_run_smoke "${tag}" "${log_file}"
  local rc=$?
  set -e
  cat "${log_file}"
  return "${rc}"
}

# ========= matrix driver =========

run_matrix() {
  local -a classifier_list os_list
  split_csv classifier_list "${CLASSIFIERS}"
  split_csv os_list "${OS_LIST}"

  local failed=0 ran=0
  local classifier os jar major
  for classifier in "${classifier_list[@]}"; do
    if [[ "${USE_CENTRAL}" -eq 0 ]]; then
      jar="$(jar_for_classifier "${classifier}")"
      if [[ ! -f "${jar}" ]]; then
        warn "Skipping missing classifier '${classifier}' (${jar})"
        continue
      fi
    fi
    major="$(cuda_major_for_classifier "${classifier}")"
    for os in "${os_list[@]}"; do
      if ! base_image_for "${major}" "${os}" >/dev/null; then
        warn "Skipping unsupported combination: classifier=${classifier} (cuda${major}) os=${os}"
        continue
      fi
      ran=$((ran + 1))
      if run_smoke "${classifier}" "${os}"; then
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

# ========= entry point =========

main() {
  parse_args "$@"

  if ! command -v docker >/dev/null 2>&1; then
    die "docker is required"
  fi

  resolve_maven_repo
  resolve_version

  CLASSIFIERS="$(classifiers_for_arch "${CUDA_VERSION}")"

  info "Mode:        ${MODE}"
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
