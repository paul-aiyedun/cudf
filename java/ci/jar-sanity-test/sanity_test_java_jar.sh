#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GPU sanity test for the Maven-repository layout produced by the java-gather
# job in .github/workflows/build.yaml (or assemble_maven_repo.sh locally).
#
# Self-contained: launches a throwaway RAPIDS ci-conda container (with GPU
# access) that provides the JDK + Maven toolchain, validates the gathered
# artifact layout, builds a tiny driver against the selected classifier JAR,
# and exercises basic cuDF Java operations on the GPU.
#
# No local JDK, Maven, or GPU toolkit is required on the host -- only Docker
# with the NVIDIA container runtime. Zip inputs additionally require unzip.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
CI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
. "${CI_DIR}/argparse.sh"

MAVEN_REPO_INPUT=""
MAVEN_REPO_DIR=""
CLASSIFIER=""
IMAGE=""
MAVEN_REPO_EXTRACT_DIR=""
UNZIP_CLEANUP=0

GROUP_PATH="ai/rapids/cudf"
MAIN_CLASS="ai.rapids.cudf.ci.JarSanityTest"
SANITY_VERSION="1.0-SNAPSHOT"

print_help() {
  cat << EOF

Usage: sanity_test_java_jar.sh --maven-repo-dir <path> [OPTIONS]

Runs a minimal GPU sanity test against a gathered cuDF Java Maven repository,
inside a throwaway RAPIDS ci-conda container that supplies the JDK + Maven
toolchain. The input is either the root of the cudf_java_maven_repo artifact
or a .zip archive downloaded from GitHub Actions:

    <input>/ai/rapids/cudf/<version>/
        cudf-<version>-<classifier>.jar
        cudf-<version>.pom

REQUIRED:
    -m, --maven-repo-dir Path to the gathered Maven repository root, or a
                         .zip archive containing that layout (as produced by
                         java-gather / assemble_maven_repo.sh). Zip archives
                         with a single top-level directory are accepted.

OPTIONS:
    -c, --classifier     Maven classifier to test (e.g. cuda12, cuda13,
                         cuda12-arm64). Default: auto-detect inside the
                         container from the GPU CUDA version + host arch,
                         then verify the JAR exists in the gathered repo.
    -i, --image          Container image providing conda (JDK + Maven are
                         installed into a throwaway env at runtime).
                         Default: rapidsai/ci-conda:<rapids_version>-latest.
    -h, --help           Show this help message.

REQUIREMENTS:
    Docker with the NVIDIA container runtime (nvidia-smi must work inside
    'docker run --gpus all'). No host JDK/Maven/CUDA toolkit needed.

EXAMPLES:
    # Maven repository directory:
    ./java/ci/jar-sanity-test/sanity_test_java_jar.sh --maven-repo-dir /tmp/cudf_java_maven_repo

    # GitHub Actions artifact zip:
    ./java/ci/jar-sanity-test/sanity_test_java_jar.sh -m /tmp/cudf_java_maven_repo.zip

    # Explicit classifier (e.g. when auto-detect does not match the JAR set):
    ./java/ci/jar-sanity-test/sanity_test_java_jar.sh -m /tmp/maven-repo.zip -c cuda13

EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      -h|--help)
        print_help
        exit 0
        ;;
      -m|--maven-repo-dir)
        require_value "$1" "$2"
        MAVEN_REPO_INPUT=$2
        shift 2
        ;;
      -c|--classifier)
        require_value "$1" "$2"
        CLASSIFIER=$2
        shift 2
        ;;
      -i|--image)
        require_value "$1" "$2"
        IMAGE=$2
        shift 2
        ;;
      *)
        echo "Error: Unknown argument $1"
        print_help
        exit 1
        ;;
    esac
  done
}

find_maven_repo_root() {
  local base=$1 candidate
  if [[ -d ${base}/${GROUP_PATH} ]]; then
    echo "${base}"
    return 0
  fi
  for candidate in "${base}"/*/; do
    if [[ -d ${candidate}${GROUP_PATH} ]]; then
      echo "${candidate%/}"
      return 0
    fi
  done
  echo "Error: could not find ${GROUP_PATH} under '${base}'." >&2
  exit 1
}

cleanup_extracted_repo() {
  if [[ ${UNZIP_CLEANUP} -eq 1 && -n ${MAVEN_REPO_EXTRACT_DIR} && -d ${MAVEN_REPO_EXTRACT_DIR} ]]; then
    rm -rf "${MAVEN_REPO_EXTRACT_DIR}"
  fi
}

prepare_maven_repo_input() {
  local input=$1
  if [[ ! -e ${input} ]]; then
    echo "Error: --maven-repo-dir '${input}' does not exist." >&2
    exit 1
  fi

  if [[ -d ${input} ]]; then
    MAVEN_REPO_DIR=$(find_maven_repo_root "$(cd "${input}" && pwd)")
    return
  fi

  if [[ ! -f ${input} ]]; then
    echo "Error: --maven-repo-dir '${input}' is not a directory or file." >&2
    exit 1
  fi

  case "${input}" in
    *.zip|*.ZIP) ;;
    *)
      echo "Error: --maven-repo-dir '${input}' is a file but not a .zip archive." >&2
      exit 1
      ;;
  esac

  if ! command -v unzip > /dev/null 2>&1; then
    echo "Error: 'unzip' not found on PATH (required to extract .zip input)." >&2
    exit 1
  fi

  input="$(cd "$(dirname "${input}")" && pwd)/$(basename "${input}")"
  MAVEN_REPO_EXTRACT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/cudf-java-maven-repo.XXXXXX")
  UNZIP_CLEANUP=1
  trap cleanup_extracted_repo EXIT

  echo "Extracting ${input} to ${MAVEN_REPO_EXTRACT_DIR}"
  unzip -q "${input}" -d "${MAVEN_REPO_EXTRACT_DIR}"
  MAVEN_REPO_DIR=$(find_maven_repo_root "${MAVEN_REPO_EXTRACT_DIR}")
}

detect_cuda_major() {
  local out=""
  if command -v nvcc > /dev/null 2>&1; then
    out=$(nvcc --version 2>/dev/null || true)
    if [[ ${out} =~ Cuda\ compilation\ tools,\ release\ ([0-9]+) ]]; then
      echo "${BASH_REMATCH[1]}"
      return 0
    fi
  fi
  if command -v nvidia-smi > /dev/null 2>&1; then
    out=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\).*/\1/p' | head -1)
    if [[ -n ${out} ]]; then
      echo "${out}"
      return 0
    fi
  fi
  return 1
}

detect_default_classifier() {
  local cuda_major arch_suffix=""
  cuda_major=$(detect_cuda_major) || {
    echo "Error: could not detect CUDA major version (need nvcc or nvidia-smi)." >&2
    exit 1
  }
  case "$(uname -m)" in
    x86_64|amd64) arch_suffix="" ;;
    aarch64|arm64) arch_suffix="-arm64" ;;
    *)
      echo "Error: unsupported host architecture '$(uname -m)'." >&2
      exit 1
      ;;
  esac
  echo "cuda${cuda_major}${arch_suffix}"
}

list_classifiers_in_repo() {
  local version_dir=$1 version=$2 jar base classifier
  for jar in "${version_dir}"/cudf-"${version}"-*.jar; do
    if [[ ! -f ${jar} ]]; then
      continue
    fi
    base=$(basename "${jar}" .jar)
    classifier=${base#cudf-${version}-}
    printf '%s\n' "${classifier}"
  done
}

validate_maven_repo_layout() {
  local repo_root=$1 group_dir version_dir version
  if [[ ! -d ${repo_root}/${GROUP_PATH} ]]; then
    echo "Error: expected ${repo_root}/${GROUP_PATH} in gathered Maven repo." >&2
    exit 1
  fi
  group_dir="${repo_root}/${GROUP_PATH}"
  version_dir=$(find "${group_dir}" -mindepth 1 -maxdepth 1 -type d | head -1 || true)
  if [[ -z ${version_dir} ]]; then
    echo "Error: no version directory found under ${group_dir}." >&2
    exit 1
  fi
  version=$(basename "${version_dir}")
  if [[ ! -f ${version_dir}/cudf-${version}.pom ]]; then
    echo "Error: missing POM at ${version_dir}/cudf-${version}.pom" >&2
    exit 1
  fi
  if [[ -z $(list_classifiers_in_repo "${version_dir}" "${version}") ]]; then
    echo "Error: no classifier JARs found under ${version_dir}." >&2
    exit 1
  fi
  CUDF_VERSION=${version}
  CUDF_VERSION_DIR=${version_dir}
}

select_classifier() {
  local available found=0 classifier
  available=$(list_classifiers_in_repo "${CUDF_VERSION_DIR}" "${CUDF_VERSION}")
  if [[ -z ${CLASSIFIER} ]]; then
    CLASSIFIER=$(detect_default_classifier)
    echo "Auto-detected classifier: ${CLASSIFIER}"
  fi
  while IFS= read -r classifier; do
    if [[ ${classifier} == "${CLASSIFIER}" ]]; then
      found=1
      break
    fi
  done <<< "${available}"
  if [[ ${found} -eq 0 ]]; then
    echo "Error: classifier '${CLASSIFIER}' not found in gathered repo." >&2
    echo "Available classifiers:" >&2
    while IFS= read -r classifier; do
      echo "  - ${classifier}" >&2
    done <<< "${available}"
    exit 1
  fi
  CUDF_JAR="${CUDF_VERSION_DIR}/cudf-${CUDF_VERSION}-${CLASSIFIER}.jar"
}

validate_jar_structure() {
  local jar=$1 native_count
  if ! jar tf "${jar}" > /dev/null 2>&1; then
    echo "Error: '${jar}' is not a valid JAR archive." >&2
    exit 1
  fi
  native_count=$(jar tf "${jar}" | grep -c -E '\.so$' || true)
  if [[ ${native_count} -eq 0 ]]; then
    echo "Error: '${jar}' contains no bundled native libraries (.so)." >&2
    exit 1
  fi
  echo "Validated JAR structure: ${native_count} native library entries"
}

run_in_container() {
  local sanity_project_src="${SCRIPT_DIR}"
  local work_dir project_dir m2_repo sanity_jar dep_dir classpath_str

  work_dir=$(mktemp -d /tmp/cudf-java-sanity.XXXXXX)
  project_dir="${work_dir}/jar-sanity-test"
  m2_repo="${work_dir}/m2"

  . /opt/conda/etc/profile.d/conda.sh

  rapids-logger "Generating sanity-test conda environment (jdk + maven)"
  rapids-mamba-retry create --yes -n sanity_java maven "openjdk=8.*"
  conda activate sanity_java

  echo "cuDF Java JAR sanity test (in container)"
  echo "  maven repo: ${MAVEN_REPO_DIR}"
  echo "  repo root:  ${REPO_ROOT}"
  echo "  work dir:   ${work_dir}"
  echo "  java:       $(java -version 2>&1 | head -1)"
  echo "  mvn:        $(mvn --version 2>&1 | head -1)"

  validate_maven_repo_layout "${MAVEN_REPO_DIR}"
  select_classifier
  validate_jar_structure "${CUDF_JAR}"

  echo "Using cudf version ${CUDF_VERSION}, classifier ${CLASSIFIER}"
  echo "  jar: ${CUDF_JAR}"
  echo "  pom: ${CUDF_VERSION_DIR}/cudf-${CUDF_VERSION}.pom"

  mkdir -p "${m2_repo}"
  cp -a "${MAVEN_REPO_DIR}/." "${m2_repo}/"

  cp -a "${sanity_project_src}" "${project_dir}"

  rapids-logger "Building sanity-test driver against cudf-${CUDF_VERSION}-${CLASSIFIER}.jar"
  (
    cd "${project_dir}"
    mvn -B -q package \
      -Dmaven.repo.local="${m2_repo}" \
      -Dcudf.version="${CUDF_VERSION}" \
      -Dcuda.classifier="${CLASSIFIER}"
  )

  sanity_jar="${project_dir}/target/cudf-jar-sanity-test-${SANITY_VERSION}.jar"
  dep_dir="${project_dir}/target/dependency"
  classpath_str="${sanity_jar}:${dep_dir}/*"

  echo
  rapids-logger "Running JarSanityTest with classifier ${CLASSIFIER}"
  echo "  classpath: ${classpath_str}"
  echo

  java -cp "${classpath_str}" "${MAIN_CLASS}"

  echo
  echo "Sanity test completed successfully."
}

run_on_host() {
  if ! command -v docker > /dev/null 2>&1; then
    echo "Error: 'docker' not found on PATH." >&2
    exit 1
  fi

  prepare_maven_repo_input "${MAVEN_REPO_INPUT}"

  if [[ -z ${IMAGE} ]]; then
    local rapids_version
    rapids_version="$(head -1 "${REPO_ROOT}/VERSION" | cut -d. -f1,2)"
    IMAGE="rapidsai/ci-conda:${rapids_version}-latest"
  fi

  echo "cuDF Java JAR sanity test"
  echo "  image:      ${IMAGE}"
  echo "  repo root:  ${REPO_ROOT}"
  echo "  input:      ${MAVEN_REPO_INPUT}"
  echo "  maven repo: ${MAVEN_REPO_DIR}"
  if [[ -n ${CLASSIFIER} ]]; then
    echo "  classifier: ${CLASSIFIER} (explicit)"
  else
    echo "  classifier: (auto-detect in container)"
  fi

  docker run \
    --rm \
    --gpus all \
    --volume "${REPO_ROOT}:/repo:ro" \
    --volume "${MAVEN_REPO_DIR}:/maven-repo:ro" \
    --workdir /repo \
    --env SANITY_TEST_IN_CONTAINER=1 \
    --env MAVEN_REPO_DIR=/maven-repo \
    --env REPO_ROOT=/repo \
    --env CLASSIFIER="${CLASSIFIER}" \
    "${IMAGE}" \
    bash /repo/java/ci/jar-sanity-test/sanity_test_java_jar.sh
}

if [[ ${SANITY_TEST_IN_CONTAINER:-} == 1 ]]; then
  MAVEN_REPO_DIR="${MAVEN_REPO_DIR:-/maven-repo}"
  REPO_ROOT="${REPO_ROOT:-/repo}"
  run_in_container
  exit 0
fi

parse_args "$@"
require_arg --maven-repo-dir "${MAVEN_REPO_INPUT}"
run_on_host
