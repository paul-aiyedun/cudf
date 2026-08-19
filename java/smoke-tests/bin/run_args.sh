#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Flag parsing for run.sh. Source, do not execute.

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
+ Rocky Linux 8).

Source modes (exactly one required):
  --artifact-url URL      Fetch a GitHub Actions artifact (java-build,
                          java-gather, or signed Maven Central bundle)
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
