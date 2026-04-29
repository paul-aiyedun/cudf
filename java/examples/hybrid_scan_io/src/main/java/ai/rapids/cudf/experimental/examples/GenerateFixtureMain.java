/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.experimental.examples;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ParquetWriterOptions;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.TableWriter;

import java.io.File;
import java.util.stream.IntStream;

/**
 * Writes a small deterministic Parquet file that the other examples can consume.
 *
 * <p>Mirrors the helper used by {@code HybridScanReaderTest#writeFixtureParquet}: three
 * row groups of 1000 rows each with three int columns ({@code id}, {@code zip},
 * {@code num_units}) and PAGE-level statistics. Forcing multiple row groups makes the
 * file useful both for the two-step / filter-then-payload example and for the chunked
 * pipeline example.
 *
 * <p>Usage:
 * <pre>
 * mvn -pl java/examples/hybrid_scan_io exec:java \
 *     -Dexec.mainClass=ai.rapids.cudf.experimental.examples.GenerateFixtureMain \
 *     -Dexec.args="/tmp/fixture.parquet"
 * </pre>
 */
public final class GenerateFixtureMain {
  private GenerateFixtureMain() {}

  public static void main(String[] args) {
    if (args.length < 1) {
      System.err.println("Usage: GenerateFixtureMain <output-parquet-path>");
      System.exit(1);
    }
    File out = new File(args[0]);
    File parent = out.getAbsoluteFile().getParentFile();
    if (parent != null && !parent.exists() && !parent.mkdirs()) {
      System.err.println("Could not create parent directory: " + parent);
      System.exit(2);
    }

    int rowsPerGroup = 1000;
    int numGroups = 3;
    int totalRows = rowsPerGroup * numGroups;

    boolean weInitializedRmm = false;
    try {
      if (!Rmm.isInitialized()) {
        Rmm.initialize(RmmAllocationMode.POOL, Rmm.logToStderr(), 256L * 1024L * 1024L);
        weInitializedRmm = true;
      }

      ParquetWriterOptions opts = ParquetWriterOptions.builder()
          .withNonNullableColumns("id", "zip", "num_units")
          .withRowGroupSizeRows(rowsPerGroup)
          .withStatisticsFrequency(ParquetWriterOptions.StatisticsFrequency.PAGE)
          .build();

      try (TableWriter writer = Table.writeParquetChunked(opts, out)) {
        for (int g = 0; g < numGroups; g++) {
          int start = g * rowsPerGroup;
          try (ColumnVector id = ColumnVector.fromInts(
                   IntStream.range(start, start + rowsPerGroup).toArray());
               ColumnVector zip = ColumnVector.fromInts(
                   IntStream.range(start, start + rowsPerGroup)
                       .map(i -> 10000 + i * 100).toArray());
               ColumnVector numUnits = ColumnVector.fromInts(
                   IntStream.range(start, start + rowsPerGroup)
                       .map(i -> 1 + (i % 3)).toArray());
               Table t = new Table(id, zip, numUnits)) {
            writer.write(t);
          }
        }
      }
      System.out.printf("Wrote %s (%d rows in %d row groups)%n", out, totalRows, numGroups);
    } finally {
      if (weInitializedRmm && Rmm.isInitialized()) {
        Rmm.shutdown();
      }
    }
  }
}
