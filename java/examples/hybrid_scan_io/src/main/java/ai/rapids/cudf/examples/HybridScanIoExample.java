/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.examples;

import ai.rapids.cudf.ByteRange;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.HybridScanReader;
import ai.rapids.cudf.ParquetOptions;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.UseDataPageMask;
import ai.rapids.cudf.ast.BinaryOperation;
import ai.rapids.cudf.ast.BinaryOperator;
import ai.rapids.cudf.ast.ColumnReference;
import ai.rapids.cudf.ast.CompiledExpression;
import ai.rapids.cudf.ast.Literal;

import java.io.File;
import java.io.IOException;

/**
 * Java equivalent of {@code cpp/examples/hybrid_scan_io/hybrid_scan_io.cpp}.
 *
 * <p>Reads a Parquet file three times for an apples-to-apples comparison of the legacy
 * reader and two flavours of the experimental {@link HybridScanReader}:
 * <ol>
 *   <li>{@link Table#readParquet(ParquetOptions, File)} (legacy) — read all projected
 *       columns then apply the filter on the result;</li>
 *   <li>{@code HybridScanReader} two-step with {@code RowMaskKind.ALL_TRUE} — row-group
 *       stats pruning + filter materialisation, but no page-index pruning;</li>
 *   <li>{@code HybridScanReader} two-step with {@code RowMaskKind.PAGE_INDEX_STATS} —
 *       same as (2) plus a page-index-seeded row mask consumed via
 *       {@code UseDataPageMask.YES}.</li>
 * </ol>
 * Each scenario is implemented in its own helper method and tagged with a labelled-bullet
 * prefix in stdout: {@code [Setup]}, {@code [Legacy]}, {@code [Hybrid]}, and
 * {@code [Hybrid w PageIndex]}.
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
        System.out.println("[Setup] Initialising RMM (512 MB POOL)...");
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

        long legacyTotal = runLegacy(path, readOpts, filter, columnName, literalValue);
        System.out.println();
        runHybridTwoStep(path, readOpts, filter, columnName, literalValue, legacyTotal);
        System.out.println();
        runHybridWithPageIndex(path, readOpts, filter, columnName, literalValue, legacyTotal);
      }
    } finally {
      if (Rmm.isInitialized()) {
        Rmm.shutdown();
      }
    }
  }

  /**
   * Scenario 1: legacy reader path. Reads all projected columns then applies the filter on
   * the materialised result. We deliberately do NOT push the filter into
   * {@link Table#readParquet(ParquetOptions, File, CompiledExpression)} -- the legacy
   * reader's row-group pruning is therefore not exercised. The hybrid scenarios below DO
   * prune row groups via {@code filterRowGroupsWithStats}, so the comparison is biased
   * toward hybrid by exactly one row-group's worth of decode + I/O. Treat the legacy
   * timing as an upper bound on what the two-step / page-index variants replace.
   *
   * @return the unfiltered row count of the file (reused as the {@code / N} denominator
   *         in the hybrid scenarios so all three paths print "{filtered} / {total}").
   */
  private static long runLegacy(File path, ParquetOptions readOpts, CompiledExpression filter,
                                String columnName, int literalValue) {
    System.out.println(
        "[Legacy] Reading entire file via Table.readParquet (no filter pushdown)...");
    long legacyRows;
    long legacyTotal;
    long t0 = System.nanoTime();
    try (Table legacyTable = Table.readParquet(readOpts, path);
         ColumnVector mask = filter.computeColumn(legacyTable);
         Table filtered = legacyTable.filter(mask)) {
      legacyTotal = legacyTable.getRowCount();
      legacyRows = filtered.getRowCount();
    }
    long legacyMs = (System.nanoTime() - t0) / 1_000_000L;
    System.out.printf("[Legacy] Applied filter '%s > %d' on materialised table.%n",
        columnName, literalValue);
    System.out.printf("[Legacy] %d / %d rows survive.%n", legacyRows, legacyTotal);
    System.out.printf("[Legacy] Processing time: %d ms.%n", legacyMs);
    return legacyTotal;
  }

  /**
   * Scenario 2: hybrid scan two-step with an all-true row mask. Reads only the footer up
   * front, prunes row groups via column-chunk statistics, then copies just the surviving
   * filter + payload column chunks to the device. {@code RowMaskKind.ALL_TRUE} +
   * {@code UseDataPageMask.NO} means no page-index pruning happens here -- that's what
   * scenario 3 below adds.
   */
  private static void runHybridTwoStep(File path, ParquetOptions readOpts,
                                       CompiledExpression filter, String columnName,
                                       int literalValue, long legacyTotal) throws IOException {
    System.out.println("[Hybrid] Reading just the Parquet footer (no full-file IO)...");
    long t1 = System.nanoTime();
    try (HostMemoryBuffer footer = Util.readFooterOnly(path);
         HybridScanReader reader = new HybridScanReader(footer, readOpts, filter)) {
      System.out.printf("[Hybrid] Opened HybridScanReader; footer is %d bytes.%n",
          footer.getLength());
      int[] all = reader.allRowGroups();
      int[] survived = reader.filterRowGroupsWithStats(all);
      System.out.printf(
          "[Hybrid] Stats-based row-group pruning: %d -> %d row groups survive.%n",
          all.length, survived.length);

      if (survived.length == 0) {
        System.out.println(
            "[Hybrid] All row groups pruned by statistics; nothing to read.");
        return;
      }

      ByteRange[] filterRanges = reader.filterColumnChunksByteRanges(survived);
      ByteRange[] payloadRanges = reader.payloadColumnChunksByteRanges(survived);
      System.out.printf(
          "[Hybrid] Copying %d filter column byte range(s) (%s) to device.%n",
          filterRanges.length, columnName);
      System.out.printf(
          "[Hybrid] Copying %d payload column byte range(s) to device.%n",
          payloadRanges.length);

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
        System.out.printf(
            "[Hybrid] Materialised filter columns: %d rows survive %s > %d.%n",
            fr.table().getRowCount(), columnName, literalValue);
        System.out.printf(
            "[Hybrid] Materialised payload columns aligned to row mask: %d rows.%n",
            hybridRows);
      } finally {
        Util.closeAll(filterCols);
        Util.closeAll(payloadCols);
      }
      long hybridMs = (System.nanoTime() - t1) / 1_000_000L;
      System.out.printf("[Hybrid] Total: %d / %d rows survive.%n", hybridRows, legacyTotal);
      System.out.printf("[Hybrid] Processing time: %d ms.%n", hybridMs);
    }
  }

  /**
   * Scenario 3: hybrid scan two-step with a page-index-seeded row mask. Differs from
   * scenario 2 in two ways:
   * <ul>
   *   <li>Reads the full file once up front and slices the footer + page index + chunk
   *       byte ranges from the same buffer (the page-index variant needs page-index
   *       bytes alongside chunks, so a footer-only pre-fetch would just defer the read).
   *       The {@code [Hybrid]} timer above stays footer-first to keep that variant's
   *       "skip the full file until you really need it" advantage visible in stdout.</li>
   *   <li>Materialises filter + payload columns with
   *       {@code RowMaskKind.PAGE_INDEX_STATS} + {@code UseDataPageMask.YES}, so the row
   *       mask is seeded from page-level statistics and the materialise calls actually
   *       consume the resulting data-page mask (skipping pages whose stats can't satisfy
   *       the filter).</li>
   * </ul>
   * If the file has no page index (e.g. tiny fixtures, even with PAGE statistics enabled),
   * the variant prints a notice and returns -- mirroring the hedge in
   * {@code testPageIndexByteRangeAndSetup}.
   */
  private static void runHybridWithPageIndex(File path, ParquetOptions readOpts,
                                             CompiledExpression filter, String columnName,
                                             int literalValue, long legacyTotal)
      throws IOException {
    System.out.println(
        "[Hybrid w PageIndex] Reading full file (need page index bytes alongside chunks)...");
    long t = System.nanoTime();
    try (HostMemoryBuffer file = Util.readFileToHostBuffer(path);
         HostMemoryBuffer footer = Util.extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer, readOpts, filter)) {
      System.out.printf("[Hybrid w PageIndex] Opened HybridScanReader; footer is %d bytes.%n",
          footer.getLength());

      ByteRange piRange = reader.pageIndexByteRange();
      if (piRange.size() == 0) {
        System.out.println(
            "[Hybrid w PageIndex] File has no page index; skipping this variant.");
        return;
      }
      try (HostMemoryBuffer pi = file.slice(piRange.offset(), piRange.size())) {
        reader.setupPageIndex(pi);
      }
      System.out.printf("[Hybrid w PageIndex] Loaded page index (%d bytes).%n", piRange.size());

      int[] all = reader.allRowGroups();
      int[] survived = reader.filterRowGroupsWithStats(all);
      System.out.printf(
          "[Hybrid w PageIndex] Stats-based row-group pruning: %d -> %d row groups survive.%n",
          all.length, survived.length);
      if (survived.length == 0) {
        System.out.println(
            "[Hybrid w PageIndex] All row groups pruned by statistics; nothing to read.");
        return;
      }

      ByteRange[] filterRanges  = reader.filterColumnChunksByteRanges(survived);
      ByteRange[] payloadRanges = reader.payloadColumnChunksByteRanges(survived);
      System.out.printf(
          "[Hybrid w PageIndex] Copying %d filter column byte range(s) (%s) to device.%n",
          filterRanges.length, columnName);
      System.out.printf(
          "[Hybrid w PageIndex] Copying %d payload column byte range(s) to device.%n",
          payloadRanges.length);

      DeviceMemoryBuffer[] filterCols = null;
      DeviceMemoryBuffer[] payloadCols = null;
      long hybridRows;
      try {
        filterCols  = Util.copyRangesToDevice(file, filterRanges);
        payloadCols = Util.copyRangesToDevice(file, payloadRanges);
        try (HybridScanReader.FilterMaterializationResult fr =
                 reader.materializeFilterColumns(survived, filterCols, UseDataPageMask.YES,
                     HybridScanReader.RowMaskKind.PAGE_INDEX_STATS);
             Table pTable = reader.materializePayloadColumns(survived, payloadCols,
                 fr.rowMask(), UseDataPageMask.YES)) {
          hybridRows = pTable.getRowCount();
          System.out.printf(
              "[Hybrid w PageIndex] Materialised filter columns (page-index seeded mask):"
                  + " %d rows survive %s > %d.%n",
              fr.table().getRowCount(), columnName, literalValue);
          System.out.printf(
              "[Hybrid w PageIndex] Materialised payload columns aligned to row mask: %d rows.%n",
              hybridRows);
        }
      } finally {
        Util.closeAll(filterCols);
        Util.closeAll(payloadCols);
      }
      long ms = (System.nanoTime() - t) / 1_000_000L;
      System.out.printf("[Hybrid w PageIndex] Total: %d / %d rows survive.%n",
          hybridRows, legacyTotal);
      System.out.printf("[Hybrid w PageIndex] Processing time: %d ms.%n", ms);
    }
  }
}
