/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.rapids.cudf.ast.BinaryOperation;
import ai.rapids.cudf.ast.BinaryOperator;
import ai.rapids.cudf.ast.ColumnNameReference;
import ai.rapids.cudf.ast.CompiledExpression;
import ai.rapids.cudf.ast.Literal;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * End-to-end tests for {@link HybridScanReader}.
 *
 * <p>Tests are organised in the same order as the public-API sections in
 * {@link HybridScanReader}: constructor, metadata, row-group enumeration, row-group
 * filtering, byte ranges, single-shot materialisation, chunked materialisation,
 * AST extensions, and lifecycle / error handling.
 *
 * <p>All Parquet fixtures are generated on the fly; no pre-baked {@code .parquet} file
 * is needed.
 */
public class HybridScanReaderTest extends CudfTestBase {

  // --------------------------------------------------------------------
  // Tests: HybridScanReader (constructor)
  // --------------------------------------------------------------------

  /** Verifies that passing a null footer buffer to the constructor throws IllegalArgumentException. */
  @Test
  void testNullFooterThrows() {
    assertThrows(IllegalArgumentException.class,
        () -> new HybridScanReader(null, optsForColumns("id"), null));
  }

  // --------------------------------------------------------------------
  // Tests: pageIndexByteRange() / setupPageIndex()
  // --------------------------------------------------------------------

  /** Verifies pageIndexByteRange() returns a non-zero range and setupPageIndex() ingests it without error when the file was written with COLUMN-level statistics. */
  @Test
  void testSetupPageIndexPresent(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writePageIndexParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      ByteRange piRange = reader.pageIndexByteRange();
      assertNotNull(piRange);
      assertTrue(piRange.size() > 0,
          "A file written with COLUMN-level statistics must contain a page index");
      try (HostMemoryBuffer pi = file.slice(piRange.offset(), piRange.size())) {
        assertDoesNotThrow(() -> reader.setupPageIndex(pi));
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: allRowGroups() / totalRowsInRowGroups()
  // --------------------------------------------------------------------

  /** Verifies that allRowGroups() returns all indices and totalRowsInRowGroups() sums them correctly. */
  @Test
  void testCreateAndEnumerateRowGroups(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    int rows = writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      int[] rgs = reader.allRowGroups();
      assertNotNull(rgs);
      assertTrue(rgs.length > 0);
      assertEquals(rows, reader.totalRowsInRowGroups(rgs));
    }
  }

  // --------------------------------------------------------------------
  // Tests: resetColumnSelection()
  // --------------------------------------------------------------------

  /** Verifies resetColumnSelection() does not throw after byte-range resolution has already cached the column split. */
  @Test
  void testResetColumnSelection(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      int[] rgs = reader.allRowGroups();
      reader.payloadColumnChunksByteRanges(rgs);
      assertDoesNotThrow(reader::resetColumnSelection);
    }
  }

  // --------------------------------------------------------------------
  // Tests: filterRowGroupsWithStats()
  // --------------------------------------------------------------------

  /** Verifies filterRowGroupsWithStats() prunes at least one row group when the filter threshold excludes a whole group. */
  @Test
  void testFilterRowGroupsWithStats(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    // zip_code values across the file are 10000, 10100, ..., 309900. Row groups split at 1000
    // rows each, so the second and third row groups have all-larger zip_code values. Use a
    // threshold that prunes the first row group.
    ColumnNameReference zipCol = new ColumnNameReference("zip_code");
    Literal lit = Literal.ofInt(150000);
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER, zipCol, lit);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         CompiledExpression filter = expr.compile();
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), filter)) {
      int[] all = reader.allRowGroups();
      int[] filtered = reader.filterRowGroupsWithStats(all);
      assertTrue(filtered.length <= all.length);
      if (all.length > 1) {
        assertTrue(filtered.length < all.length,
            "Stats filter should prune at least one row group when " + all.length +
            " input row groups have disjoint zip_code ranges, got " + filtered.length);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: secondaryFiltersByteRanges()
  // --------------------------------------------------------------------

  /** Verifies secondaryFiltersByteRanges() returns a structurally valid SecondaryFilterRanges even when the file has no applicable bloom or dictionary pages. */
  @Test
  void testSecondaryFiltersByteRangesShape(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    ColumnNameReference zipCol = new ColumnNameReference("zip_code");
    Literal lit = Literal.ofInt(150000);
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER, zipCol, lit);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         CompiledExpression filter = expr.compile();
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), filter)) {
      int[] rgs = reader.allRowGroups();
      SecondaryFilterRanges sfr = reader.secondaryFiltersByteRanges(rgs);
      assertNotNull(sfr);
      assertNotNull(sfr.bloomFilterRanges());
      assertNotNull(sfr.dictionaryPageRanges());
    }
  }

  // --------------------------------------------------------------------
  // Tests: filterRowGroupsWithDictionaryPages()
  // --------------------------------------------------------------------

  /** Verifies filterRowGroupsWithDictionaryPages() completes without error and does not expand the input row-group set. */
  @Test
  void testFilterRowGroupsWithDictionaryPages(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    ColumnNameReference zipCol = new ColumnNameReference("zip_code");
    Literal lit = Literal.ofInt(50000);
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER, zipCol, lit);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         CompiledExpression filter = expr.compile();
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), filter)) {
      int[] rgs = reader.allRowGroups();
      SecondaryFilterRanges sfr = reader.secondaryFiltersByteRanges(rgs);
      // Dictionary page ranges may be empty for high-cardinality integer columns; the
      // call must still succeed and must not expand the input row-group set.
      ByteRange[] dictRanges = sfr.dictionaryPageRangesArray();
      DeviceMemoryBuffer[] dictBufs = copyRangesToDevice(file, dictRanges);
      try {
        int[] result = reader.filterRowGroupsWithDictionaryPages(dictBufs, rgs);
        assertNotNull(result);
        assertTrue(result.length <= rgs.length,
            "filterRowGroupsWithDictionaryPages must not expand the row-group set");
      } finally {
        closeAll(dictBufs);
      }
    }
  }

  // TODO: add testFilterRowGroupsWithBloomFilters once ParquetWriterOptions exposes
  //       bloom filter writing (set_column_chunks_bloom_filter_params). See
  //       HybridScanReader.java for details.

  // --------------------------------------------------------------------
  // Tests: filterColumnChunksByteRanges() / payloadColumnChunksByteRanges()
  // --------------------------------------------------------------------

  /** Verifies payloadColumnChunksByteRanges() returns a non-empty set of ranges when no filter is set (all columns are payload). */
  @Test
  void testPayloadByteRangesNoFilter(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      int[] rgs = reader.allRowGroups();
      ByteRange[] ranges = reader.payloadColumnChunksByteRanges(rgs);
      assertNotNull(ranges);
      assertTrue(ranges.length > 0);
    }
  }

  /** Verifies that filterColumnChunksByteRanges() returns one range per row group for the filter column and payloadColumnChunksByteRanges() returns one range per payload column per row group. */
  @Test
  void testFilterAndPayloadByteRangeSplit(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    ColumnNameReference zipCol = new ColumnNameReference("zip_code");
    Literal lit = Literal.ofInt(50000);
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER, zipCol, lit);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         CompiledExpression filter = expr.compile();
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), filter)) {
      int[] rgs = reader.allRowGroups();
      ByteRange[] filterRanges = reader.filterColumnChunksByteRanges(rgs);
      ByteRange[] payloadRanges = reader.payloadColumnChunksByteRanges(rgs);
      assertNotNull(filterRanges);
      assertNotNull(payloadRanges);
      // With one filter column ("zip_code") and two payload columns ("id", "num_units"), the
      // filter ranges should equal one chunk per row group and payload ranges two per group.
      assertEquals(rgs.length, filterRanges.length);
      assertEquals(rgs.length * 2, payloadRanges.length);
    }
  }

  // --------------------------------------------------------------------
  // Tests: allColumnChunksByteRanges()
  //   Exercised as a setup step in testMaterializeAllColumns and testChunkedAllColumns below.
  // --------------------------------------------------------------------

  // --------------------------------------------------------------------
  // Tests: materializeFilterColumns() / materializePayloadColumns()
  // --------------------------------------------------------------------

  /** Verifies the explicit two-step flow: materializeFilterColumns + materializePayloadColumns produce tables with matching row counts. */
  @Test
  void testExplicitTwoStepFlow(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    ColumnNameReference zipCol = new ColumnNameReference("zip_code");
    Literal lit = Literal.ofInt(50000);
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER, zipCol, lit);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         CompiledExpression filter = expr.compile();
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), filter)) {
      int[] rgs = reader.allRowGroups();
      int[] survived = reader.filterRowGroupsWithStats(rgs);
      if (survived.length == 0) {
        // Filter pruned everything; nothing to materialize. Test still validates the API.
        return;
      }
      DeviceMemoryBuffer[] filterCols = null;
      DeviceMemoryBuffer[] payloadCols = null;
      try {
        ByteRange[] filterRanges = reader.filterColumnChunksByteRanges(survived);
        ByteRange[] payloadRanges = reader.payloadColumnChunksByteRanges(survived);
        filterCols = copyRangesToDevice(file, filterRanges);
        payloadCols = copyRangesToDevice(file, payloadRanges);
        try (HybridScanReader.FilterMaterializationResult fr =
                 reader.materializeFilterColumns(survived, filterCols, UseDataPageMask.NO,
                     HybridScanReader.RowMaskKind.ALL_TRUE);
             Table payload = reader.materializePayloadColumns(survived, payloadCols,
                 fr.rowMask(), UseDataPageMask.NO)) {
          assertEquals(fr.table().getRowCount(), payload.getRowCount());
        }
      } finally {
        closeAll(filterCols);
        closeAll(payloadCols);
      }
    }
  }

  /**
   * Verifies materializeFilterColumns with PAGE_INDEX_STATS + UseDataPageMask.YES produces
   * the same surviving row count as the ALL_TRUE variant.
   */
  @Test
  void testMaterializeFilterColumnsWithPageIndexStats(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writePageIndexParquet(pq);
    // Filter that prunes groups 0 and 1 entirely by row-group stats (their max zip_code is
    // 4,999 and 54,999 respectively, both below 99,999), leaving only group 2 (100,000–104,999).
    ColumnNameReference zipCol = new ColumnNameReference("zip_code");
    Literal lit = Literal.ofInt(99999);
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER, zipCol, lit);

    // Reference: ALL_TRUE, no page index.
    long refRowCount;
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         CompiledExpression filter = expr.compile();
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), filter)) {
      int[] rgs = reader.allRowGroups();
      int[] survived = reader.filterRowGroupsWithStats(rgs);
      if (survived.length == 0) {
        refRowCount = 0;
      } else {
        DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
            file, reader.filterColumnChunksByteRanges(survived));
        DeviceMemoryBuffer[] payloadCols = copyRangesToDevice(
            file, reader.payloadColumnChunksByteRanges(survived));
        try (HybridScanReader.FilterMaterializationResult fr =
                 reader.materializeFilterColumns(survived, filterCols, UseDataPageMask.NO,
                     HybridScanReader.RowMaskKind.ALL_TRUE);
             Table payload = reader.materializePayloadColumns(survived, payloadCols,
                 fr.rowMask(), UseDataPageMask.NO)) {
          refRowCount = payload.getRowCount();
        } finally {
          closeAll(filterCols);
          closeAll(payloadCols);
        }
      }
    }

    // PAGE_INDEX_STATS path: page index must be present and must give the same result.
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         CompiledExpression filter = expr.compile();
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), filter)) {
      ByteRange piRange = reader.pageIndexByteRange();
      assertTrue(piRange.size() > 0, "Page index must be present for this test");
      try (HostMemoryBuffer pi = file.slice(piRange.offset(), piRange.size())) {
        reader.setupPageIndex(pi);
      }
      int[] rgs = reader.allRowGroups();
      int[] survived = reader.filterRowGroupsWithStats(rgs);
      if (survived.length == 0) {
        assertEquals(0, refRowCount,
            "PAGE_INDEX_STATS must produce the same row count as ALL_TRUE");
        return;
      }
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          file, reader.filterColumnChunksByteRanges(survived));
      DeviceMemoryBuffer[] payloadCols = copyRangesToDevice(
          file, reader.payloadColumnChunksByteRanges(survived));
      try (HybridScanReader.FilterMaterializationResult fr =
               reader.materializeFilterColumns(survived, filterCols, UseDataPageMask.YES,
                   HybridScanReader.RowMaskKind.PAGE_INDEX_STATS);
           Table payload = reader.materializePayloadColumns(survived, payloadCols,
               fr.rowMask(), UseDataPageMask.YES)) {
        assertEquals(refRowCount, payload.getRowCount(),
            "PAGE_INDEX_STATS must produce the same row count as ALL_TRUE");
      } finally {
        closeAll(filterCols);
        closeAll(payloadCols);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: materializeAllColumns()
  // --------------------------------------------------------------------

  /** Verifies materializeAllColumns returns a table with all projected columns and the correct total row count. */
  @Test
  void testMaterializeAllColumns(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    int rows = writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      int[] rgs = reader.allRowGroups();
      ByteRange[] ranges = reader.allColumnChunksByteRanges(rgs);
      DeviceMemoryBuffer[] devs = copyRangesToDevice(file, ranges);
      try (Table t = reader.materializeAllColumns(rgs, devs)) {
        assertEquals(3, t.getNumberOfColumns());
        assertEquals(rows, t.getRowCount());
      } finally {
        closeAll(devs);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: setupChunkingForFilterColumns() / materializeFilterColumnsChunk() /
  //        takeFilterRowMask() / setupChunkingForPayloadColumns() /
  //        materializePayloadColumnsChunk()
  // --------------------------------------------------------------------

  /**
   * Verifies the chunked filter+payload pipeline:
   * setupChunkingForFilterColumns → materializeFilterColumnsChunk → takeFilterRowMask →
   * setupChunkingForPayloadColumns → materializePayloadColumnsChunk produce the same
   * row count end-to-end.
   */
  @Test
  void testChunkedFilterColumnsPipeline(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    ColumnNameReference zipCol = new ColumnNameReference("zip_code");
    Literal lit = Literal.ofInt(50000);
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER, zipCol, lit);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         CompiledExpression filter = expr.compile();
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), filter)) {
      int[] rgs = reader.allRowGroups();
      int[] survived = reader.filterRowGroupsWithStats(rgs);
      if (survived.length == 0) {
        return;
      }
      ByteRange[] filterRanges = reader.filterColumnChunksByteRanges(survived);
      ByteRange[] payloadRanges = reader.payloadColumnChunksByteRanges(survived);
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(file, filterRanges);
      try {
        // Step 1: set up and drain chunked filter-column materialisation.
        reader.setupChunkingForFilterColumns(0L, 0L, survived,
            UseDataPageMask.NO, HybridScanReader.RowMaskKind.ALL_TRUE, filterCols);
        long filterRows = 0;
        while (reader.hasNextTableChunk()) {
          try (Table chunk = reader.materializeFilterColumnsChunk()) {
            filterRows += chunk.getRowCount();
          }
        }
        // Step 2: transfer the row mask produced by the filter pipeline to the caller.
        try (ColumnVector rowMask = reader.takeFilterRowMask()) {
          assertNotNull(rowMask);
          // Step 3: set up and drain chunked payload-column materialisation.
          DeviceMemoryBuffer[] payloadCols = copyRangesToDevice(file, payloadRanges);
          try {
            reader.setupChunkingForPayloadColumns(0L, 0L, survived,
                rowMask, UseDataPageMask.NO, payloadCols);
            long payloadRows = 0;
            while (reader.hasNextTableChunk()) {
              try (Table chunk = reader.materializePayloadColumnsChunk(rowMask)) {
                payloadRows += chunk.getRowCount();
              }
            }
            assertEquals(filterRows, payloadRows,
                "Filter and payload pipelines must yield the same row count");
          } finally {
            closeAll(payloadCols);
          }
        }
      } finally {
        closeAll(filterCols);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: setupChunkingForAllColumns() / materializeAllColumnsChunk() / hasNextTableChunk()
  // --------------------------------------------------------------------

  /** Verifies the chunked all-columns pipeline: setupChunkingForAllColumns + materializeAllColumnsChunk drain the full row count in at least one chunk. */
  @Test
  void testChunkedAllColumns(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    int rows = writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      int[] rgs = reader.allRowGroups();
      ByteRange[] ranges = reader.allColumnChunksByteRanges(rgs);
      DeviceMemoryBuffer[] devs = copyRangesToDevice(file, ranges);
      try {
        reader.setupChunkingForAllColumns(0L, 0L, rgs, devs);
        long totalRows = 0;
        int chunks = 0;
        while (reader.hasNextTableChunk()) {
          try (Table chunk = reader.materializeAllColumnsChunk()) {
            assertEquals(3, chunk.getNumberOfColumns());
            totalRows += chunk.getRowCount();
            chunks++;
          }
        }
        assertTrue(chunks >= 1);
        assertEquals(rows, totalRows);
      } finally {
        closeAll(devs);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: constructRowGroupPasses()
  // --------------------------------------------------------------------

  /** Verifies constructRowGroupPasses partitions every row group into at least one pass and the union of passes equals the input. */
  @Test
  void testConstructRowGroupPasses(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      int[] all = reader.allRowGroups();
      int[][] passes = reader.constructRowGroupPasses(all, 0L);
      assertNotNull(passes);
      assertTrue(passes.length >= 1, "Expected at least one pass");
      Set<Integer> union = new HashSet<>();
      for (int[] pass : passes) {
        for (int rg : pass) {
          union.add(rg);
        }
      }
      Set<Integer> expected = new HashSet<>();
      for (int rg : all) {
        expected.add(rg);
      }
      assertEquals(expected, union, "Union of passes must equal input row groups");
    }
  }

  // --------------------------------------------------------------------
  // Tests: ColumnNameReference (AST extension)
  // --------------------------------------------------------------------

  /** Verifies that a ColumnNameReference AST node compiles and closes cleanly via the JNI deserializer. */
  @Test
  void testAstColumnNameReferenceCompilesAndCloses() {
    ColumnNameReference c = new ColumnNameReference("zip_code");
    Literal v = Literal.ofInt(100);
    BinaryOperation e = new BinaryOperation(BinaryOperator.GREATER, c, v);
    // Compile + close should not throw; verifies the JNI deserializer accepts
    // node type 5 (COLUMN_NAME_REFERENCE).
    assertDoesNotThrow(() -> {
      try (CompiledExpression compiled = e.compile()) {
        assertNotNull(compiled);
      }
    });
  }

  // --------------------------------------------------------------------
  // Tests: close()
  // --------------------------------------------------------------------

  /** Verifies that calling close() twice on the same reader does not throw. */
  @Test
  void testCloseIsIdempotent(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      reader.close();
    }
  }

  /** Verifies that calling any reader method after close() throws IllegalStateException. */
  @Test
  void testCallAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class, reader::allRowGroups);
    }
  }

  // --------------------------------------------------------------------
  // Fixture helpers
  // --------------------------------------------------------------------

  /**
   * Writes a 3-row-group Parquet file: int columns {@code id} (globally sequential),
   * {@code zip_code} (10000 + id * 100), and {@code num_units} (1..3 cycle).
   * Uses {@code PAGE} statistics; 1,000 rows per group (3,000 total).
   *
   * @return total row count (3,000)
   */
  private static int writeFixtureParquet(File path) {
    int rowsPerGroup = 1000;
    int numGroups = 3;
    int rows = rowsPerGroup * numGroups;
    ParquetWriterOptions opts = ParquetWriterOptions.builder()
        .withNonNullableColumns("id", "zip_code", "num_units")
        .withRowGroupSizeRows(rowsPerGroup)
        .withStatisticsFrequency(ParquetWriterOptions.StatisticsFrequency.PAGE)
        .build();
    try (TableWriter writer = Table.writeParquetChunked(opts, path)) {
      for (int g = 0; g < numGroups; g++) {
        int start = g * rowsPerGroup;
        try (ColumnVector id = ColumnVector.fromInts(
                 IntStream.range(start, start + rowsPerGroup).toArray());
             ColumnVector zipCode = ColumnVector.fromInts(
                 IntStream.range(start, start + rowsPerGroup)
                     .map(i -> 10000 + i * 100).toArray());
             ColumnVector numUnits = ColumnVector.fromInts(
                 IntStream.range(start, start + rowsPerGroup)
                     .map(i -> 1 + (i % 3)).toArray());
             Table t = new Table(id, zipCode, numUnits)) {
          writer.write(t);
        }
      }
    }
    return rows;
  }

  /**
   * Writes a 3-row-group Parquet file with {@code COLUMN}-level statistics, guaranteeing
   * a non-empty page index. Each group gets a non-overlapping {@code zip_code} range
   * (group {@code g}: base = g × 50,000):
   * <ul>
   *   <li>Group 0: zip_code 0–4,999</li>
   *   <li>Group 1: zip_code 50,000–54,999</li>
   *   <li>Group 2: zip_code 100,000–104,999</li>
   * </ul>
   * 5,000 rows per group (15,000 total).
   *
   * @return total row count (15,000)
   */
  private static int writePageIndexParquet(File path) {
    int rowsPerGroup = 5_000;
    int numGroups = 3;
    int rows = rowsPerGroup * numGroups;
    int[] zipBases = {0, 50_000, 100_000};
    ParquetWriterOptions opts = ParquetWriterOptions.builder()
        .withNonNullableColumns("id", "zip_code", "num_units")
        .withRowGroupSizeRows(rowsPerGroup)
        .withStatisticsFrequency(ParquetWriterOptions.StatisticsFrequency.COLUMN)
        .build();
    try (TableWriter writer = Table.writeParquetChunked(opts, path)) {
      for (int g = 0; g < numGroups; g++) {
        int start = g * rowsPerGroup;
        int zipBase = zipBases[g];
        try (ColumnVector id = ColumnVector.fromInts(
                 IntStream.range(start, start + rowsPerGroup).toArray());
             ColumnVector zipCode = ColumnVector.fromInts(
                 IntStream.range(0, rowsPerGroup).map(i -> zipBase + i).toArray());
             ColumnVector numUnits = ColumnVector.fromInts(
                 IntStream.range(0, rowsPerGroup).map(i -> 1 + (i % 3)).toArray());
             Table t = new Table(id, zipCode, numUnits)) {
          writer.write(t);
        }
      }
    }
    return rows;
  }

  /** Read the entire file into a {@link HostMemoryBuffer}. */
  private static HostMemoryBuffer readFileToHostBuffer(File file) throws IOException {
    byte[] fileBytes = Files.readAllBytes(file.toPath());
    HostMemoryBuffer buffer = HostMemoryBuffer.allocate(fileBytes.length);
    buffer.setBytes(0, fileBytes, 0, fileBytes.length);
    return buffer;
  }

  /**
   * Extract the Parquet file footer from a host buffer.
   * Format: {@code [footer_bytes][4-byte LE footer_length][4-byte magic "PAR1"]}.
   */
  private static HostMemoryBuffer extractFooter(HostMemoryBuffer fileBuffer) {
    long fileLen = fileBuffer.getLength();
    int footerLength = fileBuffer.getInt(fileLen - 8);
    long footerStart = fileLen - 8 - footerLength;
    byte[] footerBytes = new byte[footerLength];
    fileBuffer.getBytes(footerBytes, 0, footerStart, footerLength);
    HostMemoryBuffer footer = HostMemoryBuffer.allocate(footerLength);
    footer.setBytes(0, footerBytes, 0, footerLength);
    return footer;
  }

  /** Copy byte ranges from a host buffer into device buffers (one per range). */
  private static DeviceMemoryBuffer[] copyRangesToDevice(HostMemoryBuffer fileBuffer,
                                                         ByteRange[] ranges) {
    DeviceMemoryBuffer[] out = new DeviceMemoryBuffer[ranges.length];
    try {
      for (int i = 0; i < ranges.length; i++) {
        ByteRange r = ranges[i];
        DeviceMemoryBuffer dev = DeviceMemoryBuffer.allocate(r.size());
        try (HostMemoryBuffer slice = fileBuffer.slice(r.offset(), r.size())) {
          dev.copyFromHostBuffer(slice);
        }
        out[i] = dev;
      }
      return out;
    } catch (Throwable t) {
      for (DeviceMemoryBuffer b : out) {
        if (b != null) b.close();
      }
      throw t;
    }
  }

  private static void closeAll(DeviceMemoryBuffer[] buffers) {
    if (buffers == null) return;
    for (DeviceMemoryBuffer b : buffers) {
      if (b != null) b.close();
    }
  }

  private static ParquetOptions optsForColumns(String... cols) {
    ParquetOptions.Builder b = ParquetOptions.builder();
    for (String c : cols) {
      b.includeColumn(c);
    }
    return b.build();
  }
}
