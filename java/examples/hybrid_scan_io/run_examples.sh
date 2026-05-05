#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# run_examples.sh — generates sample Parquet data, runs both hybrid-scan
# examples, then cleans up.  Must be run from java/examples/hybrid_scan_io/
# after the module has been packaged (mvn package).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_FILE="${SCRIPT_DIR}/test_data.parquet"
MVN_FLAGS="-q"   # remove -q to see full Maven output

# ── helpers ────────────────────────────────────────────────────────────────
banner() {
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "══════════════════════════════════════════════════════════════"
}

step() { echo "▶  $*"; }

# ── cleanup ────────────────────────────────────────────────────────────────
cleanup() {
    if [[ -f "${DATA_FILE}" ]]; then
        step "Cleaning up: removing ${DATA_FILE}"
        rm -f "${DATA_FILE}"
    fi
}
trap cleanup EXIT

# ── stage 1: generate sample data ─────────────────────────────────────────
banner "Stage 1 — Generate sample Parquet file"
step "Output: ${DATA_FILE}"
step "Schema: id INT, zip_code INT, num_units INT  |  3 row groups x 1000 rows"
mvn ${MVN_FLAGS} exec:java \
    -Dexec.mainClass=ai.rapids.cudf.experimental.examples.GenerateSampleParquetFileMain \
    -Dexec.args="${DATA_FILE}"
echo "✔  Sample data written."

# ── stage 2: HybridScanIoExample ──────────────────────────────────────────
banner "Stage 2 — HybridScanIoExample (legacy vs hybrid-scan two-step)"
step "Filter: zip_code > 150000"
step "Compares the legacy Table.readParquet path against the two-step hybrid scan."
mvn ${MVN_FLAGS} exec:java \
    -Dexec.mainClass=ai.rapids.cudf.experimental.examples.HybridScanIoExample \
    -Dexec.args="${DATA_FILE} zip_code 150000"
echo "✔  HybridScanIoExample complete."

# ── stage 3: HybridScanPipelineExample ────────────────────────────────────
banner "Stage 3 — HybridScanPipelineExample (chunked pipeline, no filter)"
step "Row group batch size: 0 (no limit — one pass for all row groups)"
step "Chunk size:           0 (no limit — one cuDF Table chunk per pass)"
step "Demonstrates the streaming/chunked all-columns API."
mvn ${MVN_FLAGS} exec:java \
    -Dexec.mainClass=ai.rapids.cudf.experimental.examples.HybridScanPipelineExample \
    -Dexec.args="${DATA_FILE} 0 0"
echo "✔  HybridScanPipelineExample complete."

# ── done ──────────────────────────────────────────────────────────────────
banner "All examples finished successfully"
# cleanup() runs automatically via trap EXIT
