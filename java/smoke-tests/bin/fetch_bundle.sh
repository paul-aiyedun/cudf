#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download a GitHub Actions artifact into:
#   <output-dir>/runs/<run_id>/artifacts/<artifact_id>/
# Default <output-dir>: java/smoke-tests/.cache/downloads

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_TESTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSEMBLE_SCRIPT="${SMOKE_TESTS_ROOT}/../ci/assemble_maven_repo.sh"

# shellcheck source=logging.sh
. "${SCRIPT_DIR}/logging.sh"

ARTIFACT_URL=""
OUTPUT_DIR="${SMOKE_TESTS_OUTPUT_DIR:-${SMOKE_TESTS_ROOT}/.cache/downloads}"

ARTIFACT_REPO=""
ARTIFACT_RUN_ID=""
ARTIFACT_ID=""

print_help() {
  cat << EOF
Usage: $(basename "$0") --artifact-url <url> [--output-dir DIR]

Download a GitHub Actions artifact and normalize it to a Maven-repo tree at:
  <output-dir>/runs/<run_id>/artifacts/<artifact_id>/

  --output-dir DIR   Default: ${SMOKE_TESTS_ROOT}/.cache/downloads
                     (or \$SMOKE_TESTS_OUTPUT_DIR)

Requires gh. If the destination already exists and looks valid, warns and reuses it.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) print_help; exit 0 ;;
      --artifact-url)
        require_value "$1" "${2:-}"
        ARTIFACT_URL="$2"
        shift 2
        ;;
      --output-dir)
        require_value "$1" "${2:-}"
        OUTPUT_DIR="$2"
        shift 2
        ;;
      *) die "Unknown argument: $1 (try --help)" ;;
    esac
  done
  if [[ -z "${ARTIFACT_URL}" ]]; then
    die "--artifact-url is required (try --help)"
  fi
}

parse_artifact_url() {
  local url="$1"
  if [[ "${url}" =~ github\.com/([^/]+/[^/]+)/actions/runs/([A-Za-z0-9_]+)/artifacts/([A-Za-z0-9_]+)/?$ ]]; then
    ARTIFACT_REPO="${BASH_REMATCH[1]}"
    ARTIFACT_RUN_ID="${BASH_REMATCH[2]}"
    ARTIFACT_ID="${BASH_REMATCH[3]}"
  else
    die "Could not parse artifact URL: ${url}"
  fi
}

# Returns 0 (reusable), 1 (needs fresh download), or exits on corrupt state.
reuse_if_valid() {
  local dest="$1"
  if [[ -d "${dest}/ai/rapids/cudf" ]]; then
    warn "Destination already exists; reusing: ${dest}"
    return 0
  fi
  if [[ -e "${dest}" ]]; then
    die "Destination exists but is incomplete (no ai/rapids/cudf/): ${dest}"
  fi
  return 1
}

# Unpack nested zips (e.g. maven-publish central-bundle.zip).
extract_nested_zips() {
  local dest="$1" zip
  while IFS= read -r zip; do
    unzip -q -o "${zip}" -d "${dest}"
    rm -f "${zip}"
  done < <(find "${dest}" -type f -name '*.zip' ! -name 'artifact.zip')
}

# Hoist a nested ai/rapids/cudf tree to dest/, or assemble java-build classifier dirs.
normalize_layout() {
  local dest="$1"
  extract_nested_zips "${dest}"

  local cudf_dir repo_root jar tmp
  cudf_dir="$(find "${dest}" -type d -path '*/ai/rapids/cudf' -print -quit)"
  if [[ -n "${cudf_dir}" ]]; then
    repo_root="$(cd "${cudf_dir}/../../.." && pwd)"
    dest="$(cd "${dest}" && pwd)"
    if [[ "${repo_root}" != "${dest}" ]]; then
      mv "${repo_root}/ai" "${dest}/"
    fi
  else
    jar="$(find "${dest}" -type f -name 'cudf-*-cuda*.jar' -print -quit)"
    if [[ -n "${jar}" ]]; then
      tmp="$(mktemp -d "${dest}/.assemble.XXXXXX")"
      bash "${ASSEMBLE_SCRIPT}" --jars-dir "$(dirname "$(dirname "${jar}")")" --output-dir "${tmp}"
      mv "${tmp}/ai" "${dest}/"
      rm -rf "${tmp}"
    fi
  fi

  if [[ ! -d "${dest}/ai/rapids/cudf" ]]; then
    die "No ai/rapids/cudf/ under ${dest} after download"
  fi
}

main() {
  parse_args "$@"
  command -v gh >/dev/null && command -v unzip >/dev/null || die "gh and unzip are required"
  parse_artifact_url "${ARTIFACT_URL}"

  local dest="${OUTPUT_DIR}/runs/${ARTIFACT_RUN_ID}/artifacts/${ARTIFACT_ID}"
  if reuse_if_valid "${dest}"; then
    printf '%s\n' "${dest}"
    return 0
  fi

  mkdir -p "${dest}"
  info "Downloading artifact ${ARTIFACT_ID} from ${ARTIFACT_REPO} run ${ARTIFACT_RUN_ID}"
  gh api \
    -H "Accept: application/vnd.github+json" \
    "/repos/${ARTIFACT_REPO}/actions/artifacts/${ARTIFACT_ID}/zip" \
    > "${dest}/artifact.zip"
  unzip -q -o "${dest}/artifact.zip" -d "${dest}"

  normalize_layout "${dest}"
  printf 'artifact-url:%s\n' "${ARTIFACT_URL}" > "${dest}/.source"
  info "Destination: ${dest}"
  printf '%s\n' "${dest}"
}

main "$@"
