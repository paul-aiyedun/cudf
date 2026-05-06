/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * A read-only snapshot of the Parquet file footer metadata exposed by the hybrid scan reader.
 *
 * <p>Mirrors a small subset of {@code cudf::io::parquet::FileMetaData} that is currently
 * useful from Java (file format version, total number of rows, and the {@code created_by}
 * application string). Additional fields can be added in the future without breaking
 * source compatibility.
 *
 * <p>Instances of this class are constructed only by the JNI layer
 * (see {@link HybridScanReader#parquetMetadata()}).
 *
 * <p>The APIs in this file are experimental and subject to change.
 */
@Experimental
public final class FileMetaData {
  private final int version;
  private final long numRows;
  private final String createdBy;

  /** Constructed from the JNI layer. */
  FileMetaData(int version, long numRows, String createdBy) {
    this.version = version;
    this.numRows = numRows;
    this.createdBy = createdBy == null ? "" : createdBy;
  }

  /** @return the Parquet file format version (typically 1 or 2). */
  public int version() {
    return version;
  }

  /** @return the total number of rows in the file across all row groups. */
  public long numRows() {
    return numRows;
  }

  /** @return the {@code created_by} string from the file footer (may be empty). */
  public String createdBy() {
    return createdBy;
  }

  @Override
  public String toString() {
    return "FileMetaData{version=" + version +
           ", numRows=" + numRows +
           ", createdBy='" + createdBy + "'}";
  }
}
