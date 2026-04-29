/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.experimental;

/**
 * Whether to compute and use a data page mask using the row mask to skip decompression
 * and decoding of the masked pages.
 *
 * <p>Mirrors {@code cudf::io::parquet::experimental::use_data_page_mask}.
 */
public enum UseDataPageMask {
  /** Compute and use a data page mask. */
  YES(true),
  /** Do not compute or use a data page mask. */
  NO(false);

  private final boolean nativeValue;

  UseDataPageMask(boolean nativeValue) {
    this.nativeValue = nativeValue;
  }

  /**
   * @return the underlying native boolean value of the C++ {@code use_data_page_mask} enum.
   *         Intended for internal cudf use only.
   */
  public boolean getNativeValue() {
    return nativeValue;
  }
}
