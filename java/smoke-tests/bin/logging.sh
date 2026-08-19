#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Logging and small bash helpers for java/smoke-tests/bin. Source, do not execute.

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
