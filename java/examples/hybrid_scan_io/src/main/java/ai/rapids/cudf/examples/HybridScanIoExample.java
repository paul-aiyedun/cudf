/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.examples;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.ParquetOptions;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.ast.BinaryOperation;
import ai.rapids.cudf.ast.BinaryOperator;
import ai.rapids.cudf.ast.ColumnReference;
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
 *     -Dexec.mainClass=ai.rapids.cudf.examples.HybridScanIoExample \
 *     -Dexec.args="/path/to/file.parquet col_name int_value"
 * </pre>
 */
public final class HybridScanIoExample {
  private HybridScanIoExample() {}

  public static void main(String[] args) throws IOException {
    if (args.length < 3) {
      System.err.println(
          "Usage: HybridScanIoExample <parquet-file> <column-name> <int-literal>\n" +
          "  parquet-file   Path to a Parquet file" +
              " (use GenerateSampleParquetFileMain to create one)\n" +
          "  column-name    Name of an integer column to filter on (e.g. zip_code)\n" +
          "  int-literal    Integer threshold; rows where column-name > int-literal are kept");
      System.exit(1);
    }
    File path = new File(args[0]);
    String columnName = args[1];
    int literalValue = Integer.parseInt(args[2]);

    if (!path.isFile()) {
      System.err.println("Input file not found: " + path);
      System.exit(2);
    }

    // Projected columns, in the order they will appear in every materialised Table.
    // We use this list both to build the ParquetOptions and to translate the
    // user-supplied column NAME into a column INDEX for the AST filter (libcudf's
    // general AST expression parser only accepts ColumnReference, not
    // ColumnNameReference -- the latter is a hybrid-scan-reader extension).
    java.util.List<String> projected = java.util.Arrays.asList("id", "zip_code", "num_units");
    int columnIndex = projected.indexOf(columnName);
    if (columnIndex < 0) {
      System.err.println("Column must be one of " + projected + ", got: " + columnName);
      System.exit(3);
    }

    try {
      if (!Rmm.isInitialized()) {
        Rmm.initialize(RmmAllocationMode.POOL, null, 512L * 1024L * 1024L);
      }

      // Filter expression: <column-at-columnIndex> > <int-literal>.
      ColumnReference colRef = new ColumnReference(columnIndex);
      Literal lit = Literal.ofInt(literalValue);
      BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER, colRef, lit);

      try (CompiledExpression filter = expr.compile()) {
        // The hybrid scan reader splits the projected columns into two sets internally:
        //   * "filter columns"  — columns referenced by the filter expression
        //   * "payload columns" — projected columns that are NOT in the filter
        // To exercise both materialize calls, we include all columns from the fixture
        // (`id`, `zip_code`, `num_units`); the filter column is `zip_code`, leaving `id` and
        // `num_units` as the payload set. Adjust this if you point the example at
        // a different file.
        ParquetOptions.Builder optsBuilder = ParquetOptions.builder();
        for (String col : projected) {
          optsBuilder.includeColumn(col);
        }
        ParquetOptions readOpts = optsBuilder.build();

        // 1) Legacy reader path: read all projected columns then apply the filter on
        //    the result. We deliberately do NOT push the filter into Table.readParquet
        //    -- the legacy reader's row-group pruning is not exercised here. The hybrid
        //    path below still benefits from row-group pruning via
        //    filterRowGroupsWithStats, so the comparison remains representative.
        long legacyRows;
        long t0 = System.nanoTime();
        try (Table legacyTable = Table.readParquet(readOpts, path);
             ColumnVector mask = filter.computeColumn(legacyTable);
             Table filtered = legacyTable.filter(mask)) {
          legacyRows = filtered.getRowCount();
        }
        long legacyMs = (System.nanoTime() - t0) / 1_000_000L;
        System.out.printf("Legacy parquet reader: rows=%d time_ms=%d%n", legacyRows, legacyMs);

        // 2) Hybrid scan two-step (timer includes footer read and all I/O)
        long t1 = System.nanoTime();
        try (HostMemoryBuffer footer = Util.readFooterOnly(path);
             HybridScanReader reader = new HybridScanReader(footer, readOpts, filter)) {
          int[] all = reader.allRowGroups();
          int[] survived = reader.filterRowGroupsWithStats(all);
          System.out.printf("Hybrid scan: all_row_groups=%d after_stats=%d%n",
              all.length, survived.length);

          if (survived.length == 0) {
            System.out.println("Filter pruned all row groups; nothing to read.");
            return;
          }

          ByteRange[] filterRanges = reader.filterColumnChunksByteRanges(survived);
          ByteRange[] payloadRanges = reader.payloadColumnChunksByteRanges(survived);

          DeviceMemoryBuffer[] filterCols = null;
          DeviceMemoryBuffer[] payloadCols = null;
          long hybridRows;
          try (HostMemoryBuffer file = Util.readFileToHostBuffer(path)) {
            filterCols = Util.copyRangesToDevice(file, filterRanges);
            payloadCols = Util.copyRangesToDevice(file, payloadRanges);
          }
          try (HybridScanReader.FilterMaterializationResult fr =
                   reader.materializeFilterColumns(survived, filterCols, UseDataPageMask.NO,
                       HybridScanReader.RowMaskKind.ALL_TRUE);
               Table pTable = reader.materializePayloadColumns(survived, payloadCols,
                   fr.rowMask(), UseDataPageMask.NO)) {
            hybridRows = pTable.getRowCount();
            System.out.printf("Hybrid scan two-step: filter_rows=%d payload_rows=%d%n",
                fr.table().getRowCount(), hybridRows);
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
