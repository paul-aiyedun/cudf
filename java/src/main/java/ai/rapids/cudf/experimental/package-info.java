/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Experimental cudf APIs whose behavior, signatures, and on-disk formats may change
 * between releases without notice. Do not depend on these for production workloads
 * unless you accept that they may evolve in non-backward-compatible ways.
 *
 * <p>Currently the only inhabitant is the Parquet hybrid-scan reader binding
 * ({@link ai.rapids.cudf.experimental.HybridScanReader}), which mirrors
 * {@code cudf::io::parquet::experimental::hybrid_scan_reader}.
 */
package ai.rapids.cudf.experimental;
