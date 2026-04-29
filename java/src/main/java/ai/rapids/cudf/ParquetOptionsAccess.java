/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Friend-style accessor that exposes the package-private fields of
 * {@link ParquetOptions} to other cudf packages (notably
 * {@code ai.rapids.cudf.experimental}). Not intended for use by application
 * code; the regular {@link ParquetOptions} public API should be preferred.
 */
public final class ParquetOptionsAccess {
  private ParquetOptionsAccess() {}

  /** @return the list of columns to read; empty means "all columns". */
  public static String[] getIncludeColumnNames(ParquetOptions opts) {
    return opts.getIncludeColumnNames();
  }

  /** @return per-column "read binary as string" flags, in the same order as the include list. */
  public static boolean[] getReadBinaryAsString(ParquetOptions opts) {
    return opts.getReadBinaryAsString();
  }

  /** @return the configured timestamp unit, or {@link DType#EMPTY} for the file's native unit. */
  public static DType getTimeUnit(ParquetOptions opts) {
    return opts.timeUnit();
  }
}
