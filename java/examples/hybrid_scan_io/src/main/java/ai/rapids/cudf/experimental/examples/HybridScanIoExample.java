/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.experimental.examples;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.ParquetOptions;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.ast.BinaryOperation;
import ai.rapids.cudf.ast.BinaryOperator;
import ai.rapids.cudf.ast.ColumnNameReference;
import ai.rapids.cudf.ast.CompiledExpression;
import ai.rapids.cudf.ast.Literal;
import ai.rapids.cudf.experimental.ByteRange;
import ai.rapids.cudf.experimental.HybridScanReader;
import ai.rapids.cudf.experimental.UseDataPageMask;

import java.io.File;
import java.io.IOException;

/**
 * Java equivalent of {@code cpp/examples/hybrid_scan_io/hybrid_scan_io.cpp}.
 *
 * <p>Reads a Parquet file twice:
 * <ol>
 *   <li>via {@link Table#readParquet(ParquetOptions, File)} (the legacy reader);</li>
 *   <li>via the experimental {@link HybridScanReader} in two-step mode (filter columns,
 *       then payload columns);</li>
 * </ol>
 * and prints the resulting row counts so they can be compared.
 *
 * <p>Usage:
 * <pre>
 * mvn -pl java/examples/hybrid_scan_io exec:java \
 *     -Dexec.mainClass=ai.rapids.cudf.experimental.examples.HybridScanIoExample \
 *     -Dexec.args="/path/to/file.parquet col_name int_value"
 * </pre>
 */
public final class HybridScanIoExample {
  private HybridScanIoExample() {}

  public static void main(String[] args) throws IOException {
    if (args.length < 3) {
      System.err.println(
          "Usage: HybridScanIoExample <parquet-file> <column-name> <int-literal>");
      System.exit(1);
    }
    File path = new File(args[0]);
    String columnName = args[1];
    int literalValue = Integer.parseInt(args[2]);

    if (!path.isFile()) {
      System.err.println("Input file not found: " + path);
      System.exit(2);
    }

    try {
      if (!Rmm.isInitialized()) {
        Rmm.initialize(RmmAllocationMode.POOL, Rmm.logToStderr(), 512L * 1024L * 1024L);
      }

      // Filter expression: <column-name> > <int-literal>
      ColumnNameReference colRef = new ColumnNameReference(columnName);
      Literal lit = Literal.ofInt(literalValue);
      BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER, colRef, lit);

      try (CompiledExpression filter = expr.compile()) {
        // The hybrid scan reader splits the projected columns into two sets internally:
        //   * "filter columns"  — columns referenced by the filter expression
        //   * "payload columns" — projected columns that are NOT in the filter
        // To exercise both materialize calls, we include all columns from the fixture
        // (`id`, `zip`, `num_units`); the filter column is `zip`, leaving `id` and
        // `num_units` as the payload set. Adjust this if you point the example at
        // a different file.
        ParquetOptions readOpts = ParquetOptions.builder()
            .includeColumn("id")
            .includeColumn("zip")
            .includeColumn("num_units")
            .build();

        // 1) Legacy reader path
        long legacyRows;
        long t0 = System.nanoTime();
        try (Table legacyTable = Table.readParquet(readOpts, path)) {
          legacyRows = legacyTable.getRowCount();
        }
        long legacyMs = (System.nanoTime() - t0) / 1_000_000L;
        System.out.printf("Legacy parquet reader: rows=%d time_ms=%d%n", legacyRows, legacyMs);

        // 2) Hybrid scan two-step
        try (HostMemoryBuffer file = Util.readFileToHostBuffer(path);
             HostMemoryBuffer footer = Util.extractFooter(file);
             HybridScanReader reader = new HybridScanReader(footer, readOpts, filter)) {
          long t1 = System.nanoTime();
          int[] all = reader.allRowGroups();
          int[] survived = reader.filterRowGroupsWithStats(all);
          System.out.printf("Hybrid scan: all_row_groups=%d after_stats=%d%n",
              all.length, survived.length);

          if (survived.length == 0) {
            System.out.println("Filter pruned all row groups; nothing to read.");
            return;
          }

          DeviceMemoryBuffer[] filterCols = null;
          DeviceMemoryBuffer[] payloadCols = null;
          long hybridRows;
          try (ColumnVector rowMask = reader.buildAllTrueRowMask(survived)) {
            ByteRange[] filterRanges = reader.filterColumnChunksByteRanges(survived);
            ByteRange[] payloadRanges = reader.payloadColumnChunksByteRanges(survived);
            filterCols = Util.copyRangesToDevice(file, filterRanges);
            payloadCols = Util.copyRangesToDevice(file, payloadRanges);
            try (Table fTable = reader.materializeFilterColumns(survived, filterCols, rowMask,
                     UseDataPageMask.NO);
                 Table pTable = reader.materializePayloadColumns(survived, payloadCols, rowMask,
                     UseDataPageMask.NO)) {
              hybridRows = pTable.getRowCount();
              System.out.printf("Hybrid scan two-step: filter_rows=%d payload_rows=%d%n",
                  fTable.getRowCount(), hybridRows);
            }
          } finally {
            Util.closeAll(filterCols);
            Util.closeAll(payloadCols);
          }
          long hybridMs = (System.nanoTime() - t1) / 1_000_000L;
          System.out.printf("Hybrid scan total: rows=%d time_ms=%d%n", hybridRows, hybridMs);
        }
      }
    } finally {
      if (Rmm.isInitialized()) {
        Rmm.shutdown();
      }
    }
  }
}
