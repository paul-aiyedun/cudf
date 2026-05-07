/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
 * Tests for {@link HybridScanReader}.
 *
 * <p>Tests are organised in the same order as the public API: constructor,
 * {@code pageIndexByteRange()}, {@code setupPageIndex()}, {@code allRowGroups()},
 * {@code totalRowsInRowGroups()}, {@code resetColumnSelection()},
 * {@code filterRowGroupsWithStats()}, {@code secondaryFiltersByteRanges()},
 * {@code filterRowGroupsWithDictionaryPages()}, {@code filterColumnChunksByteRanges()},
 * {@code payloadColumnChunksByteRanges()}, {@code allColumnChunksByteRanges()},
 * {@code materializeFilterColumns()}, {@code materializePayloadColumns()},
 * {@code materializeAllColumns()}, the chunked-filter pipeline, the chunked-all-columns
 * pipeline, {@code hasNextTableChunk()}, {@code constructRowGroupPasses()}, and
 * {@code close()}.
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
  // Tests: pageIndexByteRange()
  // --------------------------------------------------------------------

  /**
   * Verifies pageIndexByteRange() returns a structurally valid range for a COLUMN-stats file:
   * non-zero size, positive offset, and ending exactly at the Parquet footer boundary
   * (per the spec: the page index region is contiguous and immediately precedes the footer).
   */
  @Test
  void testPageIndexByteRangeContiguousAndBeforeFooter(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writePageIndexParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      long fileLength = file.getLength();
      int footerLength = file.getInt(fileLength - 8);
      long footerStart = fileLength - 8 - footerLength;

      ByteRange piRange = reader.pageIndexByteRange();
      assertTrue(piRange.size() > 0,
          "COLUMN-stats file must contain a non-empty page index");
      assertTrue(piRange.offset() > 0,
          "Page index region must start after byte 0 (which holds the PAR1 magic)");
      assertEquals(footerStart, piRange.offset() + piRange.size(),
          "Page index region must end exactly at the Parquet footer boundary");
    }
  }

  /**
   * Verifies pageIndexByteRange() returns an empty range for a file written with ROWGROUP
   * statistics: such a file has no ColumnIndex/OffsetIndex structs and therefore no page
   * index region.
   */
  @Test
  void testPageIndexByteRangeEmptyForRowGroupStats(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeRowGroupStatsParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code"), null)) {
      ByteRange piRange = reader.pageIndexByteRange();
      assertEquals(0L, piRange.size(),
          "A ROWGROUP-stats file has no page index region");
    }
  }

  /** Verifies pageIndexByteRange() throws IllegalStateException after the reader is closed. */
  @Test
  void testPageIndexByteRangeAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class, reader::pageIndexByteRange);
    }
  }

  // --------------------------------------------------------------------
  // Tests: setupPageIndex()
  // --------------------------------------------------------------------

  /**
   * Verifies setupPageIndex() correctly populates the page-index metadata: feed it the
   * bytes returned by pageIndexByteRange(), then assert PAGE_INDEX_STATS produces the
   * exact row count expected from the fixture (group 2 alone, 5,000 rows). All 5,000
   * rows in group 2 satisfy the filter zip_code &gt; 99,999 (zip_code 100,000–104,999).
   */
  @Test
  void testSetupPageIndexPopulatesMetadata(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writePageIndexParquet(pq);
    ColumnNameReference zipCol = new ColumnNameReference("zip_code");
    Literal lit = Literal.ofInt(99999);
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER, zipCol, lit);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         CompiledExpression filter = expr.compile();
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), filter)) {
      ByteRange piRange = reader.pageIndexByteRange();
      try (HostMemoryBuffer pi = file.slice(piRange.offset(), piRange.size())) {
        reader.setupPageIndex(pi);
      }
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          file, reader.filterColumnChunksByteRanges(survived));
      DeviceMemoryBuffer[] payloadCols = copyRangesToDevice(
          file, reader.payloadColumnChunksByteRanges(survived));
      try (HybridScanReader.FilterMaterializationResult fr =
               reader.materializeFilterColumns(survived, filterCols, UseDataPageMask.YES,
                   HybridScanReader.RowMaskKind.PAGE_INDEX_STATS);
           Table payload = reader.materializePayloadColumns(survived, payloadCols,
               fr.rowMask(), UseDataPageMask.YES)) {
        assertEquals(5000L, payload.getRowCount(),
            "Group 2 (zip_code 100,000–104,999) entirely satisfies zip_code > 99,999");
      } finally {
        closeAll(filterCols);
        closeAll(payloadCols);
      }
    }
  }

  /** Verifies setupPageIndex() throws IllegalArgumentException when passed a null buffer. */
  @Test
  void testSetupPageIndexRejectsNull(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writePageIndexParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class, () -> reader.setupPageIndex(null));
    }
  }

  /**
   * Verifies that materializeFilterColumns(..., PAGE_INDEX_STATS) throws when invoked
   * without a prior setupPageIndex() call: the page-index metadata must be materialised
   * before page-level row-mask construction can succeed.
   */
  @Test
  void testPageIndexStatsRequiresSetupPageIndex(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writePageIndexParquet(pq);
    ColumnNameReference zipCol = new ColumnNameReference("zip_code");
    Literal lit = Literal.ofInt(99999);
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER, zipCol, lit);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         CompiledExpression filter = expr.compile();
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), filter)) {
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          file, reader.filterColumnChunksByteRanges(survived));
      try {
        assertThrows(CudfException.class, () ->
            reader.materializeFilterColumns(survived, filterCols, UseDataPageMask.YES,
                HybridScanReader.RowMaskKind.PAGE_INDEX_STATS));
      } finally {
        closeAll(filterCols);
      }
    }
  }

  /** Verifies setupPageIndex() throws IllegalStateException after the reader is closed. */
  @Test
  void testSetupPageIndexAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writePageIndexParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      try (HostMemoryBuffer empty = HostMemoryBuffer.allocate(1)) {
        assertThrows(IllegalStateException.class, () -> reader.setupPageIndex(empty));
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: allRowGroups()
  // --------------------------------------------------------------------

  /** Verifies allRowGroups() returns the exact contiguous indices {0, 1, 2} for a 3-group fixture. */
  @Test
  void testAllRowGroupsReturnsExactIndices(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertArrayEquals(new int[]{0, 1, 2}, reader.allRowGroups());
    }
  }

  /** Verifies allRowGroups() throws IllegalStateException after the reader is closed. */
  @Test
  void testAllRowGroupsAfterCloseThrows(@TempDir Path tmp) throws IOException {
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
  // Tests: totalRowsInRowGroups()
  // --------------------------------------------------------------------

  /** Verifies totalRowsInRowGroups() returns 3000 for all 3 groups (1000 rows × 3 groups). */
  @Test
  void testTotalRowsInRowGroupsAllGroups(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertEquals(3000L, reader.totalRowsInRowGroups(new int[]{0, 1, 2}));
    }
  }

  /** Verifies totalRowsInRowGroups() returns 1000 for a single row group. */
  @Test
  void testTotalRowsInRowGroupsSingleGroup(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertEquals(1000L, reader.totalRowsInRowGroups(new int[]{0}));
    }
  }

  /** Verifies totalRowsInRowGroups() returns 2000 for the last two row groups. */
  @Test
  void testTotalRowsInRowGroupsTwoGroups(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertEquals(2000L, reader.totalRowsInRowGroups(new int[]{1, 2}));
    }
  }

  /** Verifies totalRowsInRowGroups() throws IllegalArgumentException for a null row-group array. */
  @Test
  void testTotalRowsInRowGroupsRejectsNull(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class, () -> reader.totalRowsInRowGroups(null));
    }
  }

  /** Verifies totalRowsInRowGroups() throws IllegalStateException after the reader is closed. */
  @Test
  void testTotalRowsInRowGroupsAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class,
          () -> reader.totalRowsInRowGroups(new int[]{0}));
    }
  }

  // --------------------------------------------------------------------
  // Tests: resetColumnSelection()
  // --------------------------------------------------------------------

  /**
   * Verifies resetColumnSelection() leaves the reader in an equivalent state: byte ranges
   * resolved before and after the reset are identical (downstream observation, since the
   * reset itself has no directly observable return value).
   */
  @Test
  void testResetColumnSelectionRestoresFreshState(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      int[] rgs = reader.allRowGroups();
      ByteRange[] before = reader.payloadColumnChunksByteRanges(rgs);
      reader.resetColumnSelection();
      ByteRange[] after = reader.payloadColumnChunksByteRanges(rgs);
      assertArrayEquals(before, after,
          "resetColumnSelection() must not change which byte ranges are resolved");
    }
  }

  /** Verifies resetColumnSelection() throws IllegalStateException after the reader is closed. */
  @Test
  void testResetColumnSelectionAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class, reader::resetColumnSelection);
    }
  }

  // --------------------------------------------------------------------
  // Tests: filterRowGroupsWithStats()
  // --------------------------------------------------------------------

  /**
   * Verifies filterRowGroupsWithStats() prunes the exact expected row groups: with filter
   * zip_code &gt; 150,000, group 0 (max zip_code 109,900) is pruned and groups 1, 2 survive.
   */
  @Test
  void testFilterRowGroupsWithStatsExactSurvivors(@TempDir Path tmp) throws IOException {
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
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      assertArrayEquals(new int[]{1, 2}, survived);
    }
  }

  /** Verifies filterRowGroupsWithStats() throws IllegalArgumentException for a null row-group array. */
  @Test
  void testFilterRowGroupsWithStatsRejectsNull(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.filterRowGroupsWithStats(null));
    }
  }

  /** Verifies filterRowGroupsWithStats() throws IllegalStateException after the reader is closed. */
  @Test
  void testFilterRowGroupsWithStatsAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class,
          () -> reader.filterRowGroupsWithStats(new int[]{0}));
    }
  }

  // --------------------------------------------------------------------
  // Tests: secondaryFiltersByteRanges()
  // --------------------------------------------------------------------

  /**
   * Verifies secondaryFiltersByteRanges() returns no dictionary page ranges for a fixture
   * with high-cardinality int columns: the cuDF Parquet writer does not emit dictionary
   * pages for such columns, so the returned dictionary range array is empty.
   */
  @Test
  void testSecondaryFiltersByteRangesEmptyForHighCardinalityInts(@TempDir Path tmp) throws IOException {
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
      SecondaryFilterRanges sfr = reader.secondaryFiltersByteRanges(reader.allRowGroups());
      assertEquals(0, sfr.dictionaryPageRanges().length,
          "High-cardinality int columns must not emit dictionary pages");
    }
  }

  /** Verifies secondaryFiltersByteRanges() throws IllegalArgumentException for a null row-group array. */
  @Test
  void testSecondaryFiltersByteRangesRejectsNull(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.secondaryFiltersByteRanges(null));
    }
  }

  /** Verifies secondaryFiltersByteRanges() throws IllegalStateException after the reader is closed. */
  @Test
  void testSecondaryFiltersByteRangesAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class,
          () -> reader.secondaryFiltersByteRanges(new int[]{0}));
    }
  }

  // --------------------------------------------------------------------
  // Tests: filterRowGroupsWithDictionaryPages()
  // --------------------------------------------------------------------

  /**
   * Verifies filterRowGroupsWithDictionaryPages() returns the input row-group set unchanged
   * when no dictionary pages are present (high-cardinality int columns produce no
   * dictionaries, so there is nothing to prune against).
   */
  @Test
  void testFilterRowGroupsWithDictionaryPagesNoDictsReturnsInputUnchanged(@TempDir Path tmp) throws IOException {
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
      ByteRange[] dictRanges = sfr.dictionaryPageRanges();
      DeviceMemoryBuffer[] dictBufs = copyRangesToDevice(file, dictRanges);
      try {
        int[] result = reader.filterRowGroupsWithDictionaryPages(dictBufs, rgs);
        assertArrayEquals(rgs, result,
            "With no dictionary pages, the row-group set must be returned unchanged");
      } finally {
        closeAll(dictBufs);
      }
    }
  }

  /** Verifies filterRowGroupsWithDictionaryPages() throws IllegalArgumentException for a null buffer array. */
  @Test
  void testFilterRowGroupsWithDictionaryPagesRejectsNullBuffers(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.filterRowGroupsWithDictionaryPages(null, new int[]{0}));
    }
  }

  /** Verifies filterRowGroupsWithDictionaryPages() throws IllegalArgumentException for a null row-group array. */
  @Test
  void testFilterRowGroupsWithDictionaryPagesRejectsNullRowGroups(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.filterRowGroupsWithDictionaryPages(new DeviceMemoryBuffer[0], null));
    }
  }

  /** Verifies filterRowGroupsWithDictionaryPages() throws IllegalStateException after the reader is closed. */
  @Test
  void testFilterRowGroupsWithDictionaryPagesAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class,
          () -> reader.filterRowGroupsWithDictionaryPages(new DeviceMemoryBuffer[0], new int[]{0}));
    }
  }

  // TODO: add testFilterRowGroupsWithBloomFilters once ParquetWriterOptions exposes
  //       bloom filter writing (set_column_chunks_bloom_filter_params). See
  //       HybridScanReader.java for details.

  // --------------------------------------------------------------------
  // Tests: filterColumnChunksByteRanges()
  // --------------------------------------------------------------------

  /**
   * Verifies filterColumnChunksByteRanges() returns one range per row group for the single
   * filter column ("zip_code"): 1 column × 3 row groups = 3 ranges, each non-empty.
   */
  @Test
  void testFilterColumnChunksByteRangesOnePerGroup(@TempDir Path tmp) throws IOException {
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
      ByteRange[] ranges = reader.filterColumnChunksByteRanges(reader.allRowGroups());
      assertEquals(3, ranges.length, "1 filter column × 3 row groups");
      for (ByteRange r : ranges) {
        assertTrue(r.size() > 0, "Each filter column-chunk range must be non-empty");
      }
    }
  }

  /** Verifies filterColumnChunksByteRanges() throws IllegalArgumentException for a null row-group array. */
  @Test
  void testFilterColumnChunksByteRangesRejectsNull(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.filterColumnChunksByteRanges(null));
    }
  }

  /** Verifies filterColumnChunksByteRanges() throws IllegalStateException after the reader is closed. */
  @Test
  void testFilterColumnChunksByteRangesAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class,
          () -> reader.filterColumnChunksByteRanges(new int[]{0}));
    }
  }

  // --------------------------------------------------------------------
  // Tests: payloadColumnChunksByteRanges()
  // --------------------------------------------------------------------

  /**
   * Verifies payloadColumnChunksByteRanges() returns one range per projected column per row
   * group when no filter is set: with 3 projected columns × 3 row groups, the result has
   * 9 ranges.
   */
  @Test
  void testPayloadColumnChunksByteRangesNoFilter(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      ByteRange[] ranges = reader.payloadColumnChunksByteRanges(reader.allRowGroups());
      assertEquals(9, ranges.length, "3 payload columns × 3 row groups");
      for (ByteRange r : ranges) {
        assertTrue(r.size() > 0);
      }
    }
  }

  /**
   * Verifies payloadColumnChunksByteRanges() returns one range per projected column per row
   * group regardless of whether a filter is set: 3 columns × 3 row groups = 9. The filter
   * column is not excluded — the caller is responsible for skipping or reusing already
   * materialized filter columns.
   */
  @Test
  void testPayloadColumnChunksByteRangesWithFilter(@TempDir Path tmp) throws IOException {
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
      ByteRange[] ranges = reader.payloadColumnChunksByteRanges(reader.allRowGroups());
      assertEquals(9, ranges.length, "3 projected columns × 3 row groups (filter column not excluded)");
    }
  }

  /** Verifies payloadColumnChunksByteRanges() throws IllegalArgumentException for a null row-group array. */
  @Test
  void testPayloadColumnChunksByteRangesRejectsNull(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.payloadColumnChunksByteRanges(null));
    }
  }

  /** Verifies payloadColumnChunksByteRanges() throws IllegalStateException after the reader is closed. */
  @Test
  void testPayloadColumnChunksByteRangesAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class,
          () -> reader.payloadColumnChunksByteRanges(new int[]{0}));
    }
  }

  // --------------------------------------------------------------------
  // Tests: allColumnChunksByteRanges()
  // --------------------------------------------------------------------

  /**
   * Verifies allColumnChunksByteRanges() returns one range per projected column per row group:
   * 3 columns × 3 row groups = 9 non-empty ranges.
   */
  @Test
  void testAllColumnChunksByteRangesExactCount(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      ByteRange[] ranges = reader.allColumnChunksByteRanges(reader.allRowGroups());
      assertEquals(9, ranges.length, "3 columns × 3 row groups");
      for (ByteRange r : ranges) {
        assertTrue(r.size() > 0);
      }
    }
  }

  /** Verifies allColumnChunksByteRanges() throws IllegalArgumentException for a null row-group array. */
  @Test
  void testAllColumnChunksByteRangesRejectsNull(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.allColumnChunksByteRanges(null));
    }
  }

  /** Verifies allColumnChunksByteRanges() throws IllegalStateException after the reader is closed. */
  @Test
  void testAllColumnChunksByteRangesAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class,
          () -> reader.allColumnChunksByteRanges(new int[]{0}));
    }
  }

  // --------------------------------------------------------------------
  // Tests: materializeFilterColumns()
  // --------------------------------------------------------------------

  /**
   * Verifies materializeFilterColumns() produces a filter table with the exact expected row
   * count and column count: filter zip_code &gt; 150,000 prunes group 0 (max 109,900) and
   * keeps 599 + 1000 = 1599 rows from groups 1+2; the filter table contains the single
   * filter column ("zip_code").
   */
  @Test
  void testMaterializeFilterColumnsExactRowCount(@TempDir Path tmp) throws IOException {
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
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          file, reader.filterColumnChunksByteRanges(survived));
      try (HybridScanReader.FilterMaterializationResult fr =
               reader.materializeFilterColumns(survived, filterCols, UseDataPageMask.NO,
                   HybridScanReader.RowMaskKind.ALL_TRUE)) {
        assertEquals(1599L, fr.table().getRowCount());
        assertEquals(1, fr.table().getNumberOfColumns(), "filter table contains only zip_code");
      } finally {
        closeAll(filterCols);
      }
    }
  }

  /** Verifies materializeFilterColumns() throws IllegalArgumentException for a null row-group array. */
  @Test
  void testMaterializeFilterColumnsRejectsNullRowGroups(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.materializeFilterColumns(null, new DeviceMemoryBuffer[0],
              UseDataPageMask.NO, HybridScanReader.RowMaskKind.ALL_TRUE));
    }
  }

  /** Verifies materializeFilterColumns() throws IllegalStateException after the reader is closed. */
  @Test
  void testMaterializeFilterColumnsAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class,
          () -> reader.materializeFilterColumns(new int[]{0}, new DeviceMemoryBuffer[0],
              UseDataPageMask.NO, HybridScanReader.RowMaskKind.ALL_TRUE));
    }
  }

  // --------------------------------------------------------------------
  // Tests: materializePayloadColumns()
  // --------------------------------------------------------------------

  /**
   * Verifies materializePayloadColumns() produces a payload table with the exact expected
   * row count and column count: 1599 rows survive the filter; the payload table contains
   * the two non-filter columns ("id", "num_units").
   */
  @Test
  void testMaterializePayloadColumnsExactRowCount(@TempDir Path tmp) throws IOException {
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
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          file, reader.filterColumnChunksByteRanges(survived));
      DeviceMemoryBuffer[] payloadCols = copyRangesToDevice(
          file, reader.payloadColumnChunksByteRanges(survived));
      try (HybridScanReader.FilterMaterializationResult fr =
               reader.materializeFilterColumns(survived, filterCols, UseDataPageMask.NO,
                   HybridScanReader.RowMaskKind.ALL_TRUE);
           Table payload = reader.materializePayloadColumns(survived, payloadCols,
               fr.rowMask(), UseDataPageMask.NO)) {
        assertEquals(1599L, payload.getRowCount());
        assertEquals(2, payload.getNumberOfColumns(), "payload table contains id + num_units");
      } finally {
        closeAll(filterCols);
        closeAll(payloadCols);
      }
    }
  }

  /** Verifies materializePayloadColumns() throws IllegalArgumentException for a null row-group array. */
  @Test
  void testMaterializePayloadColumnsRejectsNullRowGroups(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null);
         ColumnVector dummyMask = ColumnVector.fromBooleans(true)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.materializePayloadColumns(null, new DeviceMemoryBuffer[0],
              dummyMask, UseDataPageMask.NO));
    }
  }

  /** Verifies materializePayloadColumns() throws IllegalArgumentException for a null row mask. */
  @Test
  void testMaterializePayloadColumnsRejectsNullRowMask(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.materializePayloadColumns(new int[]{0}, new DeviceMemoryBuffer[0],
              null, UseDataPageMask.NO));
    }
  }

  /** Verifies materializePayloadColumns() throws IllegalStateException after the reader is closed. */
  @Test
  void testMaterializePayloadColumnsAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         ColumnVector dummyMask = ColumnVector.fromBooleans(true)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class,
          () -> reader.materializePayloadColumns(new int[]{0}, new DeviceMemoryBuffer[0],
              dummyMask, UseDataPageMask.NO));
    }
  }

  // --------------------------------------------------------------------
  // Tests: materializeAllColumns()
  // --------------------------------------------------------------------

  /**
   * Verifies materializeAllColumns() returns a table with all 3 projected columns and the
   * exact total row count (3,000) for the fixture.
   */
  @Test
  void testMaterializeAllColumnsExactRowCount(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      int[] rgs = reader.allRowGroups();
      ByteRange[] ranges = reader.allColumnChunksByteRanges(rgs);
      DeviceMemoryBuffer[] devs = copyRangesToDevice(file, ranges);
      try (Table t = reader.materializeAllColumns(rgs, devs)) {
        assertEquals(3, t.getNumberOfColumns());
        assertEquals(3000L, t.getRowCount());
      } finally {
        closeAll(devs);
      }
    }
  }

  /** Verifies materializeAllColumns() throws IllegalArgumentException for a null row-group array. */
  @Test
  void testMaterializeAllColumnsRejectsNullRowGroups(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.materializeAllColumns(null, new DeviceMemoryBuffer[0]));
    }
  }

  /** Verifies materializeAllColumns() throws IllegalStateException after the reader is closed. */
  @Test
  void testMaterializeAllColumnsAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class,
          () -> reader.materializeAllColumns(new int[]{0}, new DeviceMemoryBuffer[0]));
    }
  }

  // --------------------------------------------------------------------
  // Tests: setupChunkingForFilterColumns()
  // --------------------------------------------------------------------

  /**
   * Verifies setupChunkingForFilterColumns() activates the filter-column chunked pipeline:
   * hasNextTableChunk() reports true after setup. (Calling hasNextTableChunk() before any
   * chunking has been set up is invalid in the C++ contract — see the dedicated
   * hasNextTableChunk tests.)
   */
  @Test
  void testSetupChunkingForFilterColumnsActivatesPipeline(@TempDir Path tmp) throws IOException {
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
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          file, reader.filterColumnChunksByteRanges(survived));
      try {
        reader.setupChunkingForFilterColumns(0L, 0L, survived,
            UseDataPageMask.NO, HybridScanReader.RowMaskKind.ALL_TRUE, filterCols);
        assertTrue(reader.hasNextTableChunk(), "chunking active after setup");
      } finally {
        closeAll(filterCols);
      }
    }
  }

  /** Verifies setupChunkingForFilterColumns() throws IllegalArgumentException for a null row-group array. */
  @Test
  void testSetupChunkingForFilterColumnsRejectsNullRowGroups(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.setupChunkingForFilterColumns(0L, 0L, null,
              UseDataPageMask.NO, HybridScanReader.RowMaskKind.ALL_TRUE,
              new DeviceMemoryBuffer[0]));
    }
  }

  /** Verifies setupChunkingForFilterColumns() throws IllegalStateException after the reader is closed. */
  @Test
  void testSetupChunkingForFilterColumnsAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class,
          () -> reader.setupChunkingForFilterColumns(0L, 0L, new int[]{0},
              UseDataPageMask.NO, HybridScanReader.RowMaskKind.ALL_TRUE,
              new DeviceMemoryBuffer[0]));
    }
  }

  // --------------------------------------------------------------------
  // Tests: materializeFilterColumnsChunk()
  // --------------------------------------------------------------------

  /**
   * Verifies materializeFilterColumnsChunk() drains the exact expected row count when
   * chained with setupChunkingForFilterColumns(): 1599 rows (filter zip_code &gt; 150,000
   * survives 599 + 1000 rows from groups 1+2).
   */
  @Test
  void testMaterializeFilterColumnsChunkExactTotal(@TempDir Path tmp) throws IOException {
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
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          file, reader.filterColumnChunksByteRanges(survived));
      try {
        reader.setupChunkingForFilterColumns(0L, 0L, survived,
            UseDataPageMask.NO, HybridScanReader.RowMaskKind.ALL_TRUE, filterCols);
        long total = 0;
        while (reader.hasNextTableChunk()) {
          try (Table chunk = reader.materializeFilterColumnsChunk()) {
            total += chunk.getRowCount();
          }
        }
        reader.takeFilterRowMask().close();
        assertEquals(1599L, total,
            "Filter chunks contain only the rows that survive the filter expression");
      } finally {
        closeAll(filterCols);
      }
    }
  }

  /**
   * Verifies materializeFilterColumnsChunk() throws IllegalArgumentException when invoked
   * without an active chunked filter pipeline. The JNI layer guards this state and rejects
   * the call before reaching the C++ implementation.
   */
  @Test
  void testMaterializeFilterColumnsChunkBeforeSetupThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class, reader::materializeFilterColumnsChunk);
    }
  }

  /** Verifies materializeFilterColumnsChunk() throws IllegalStateException after the reader is closed. */
  @Test
  void testMaterializeFilterColumnsChunkAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class, reader::materializeFilterColumnsChunk);
    }
  }

  // --------------------------------------------------------------------
  // Tests: takeFilterRowMask()
  // --------------------------------------------------------------------

  /**
   * Verifies takeFilterRowMask() returns a row mask whose length equals the total input row
   * count of the surviving row groups (2,000 = 1,000 × 2 surviving groups), after the
   * chunked filter pipeline has been drained.
   */
  @Test
  void testTakeFilterRowMaskAfterChunkedRun(@TempDir Path tmp) throws IOException {
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
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          file, reader.filterColumnChunksByteRanges(survived));
      try {
        reader.setupChunkingForFilterColumns(0L, 0L, survived,
            UseDataPageMask.NO, HybridScanReader.RowMaskKind.ALL_TRUE, filterCols);
        while (reader.hasNextTableChunk()) {
          reader.materializeFilterColumnsChunk().close();
        }
        try (ColumnVector rowMask = reader.takeFilterRowMask()) {
          assertEquals(2000L, rowMask.getRowCount(),
              "Row mask spans the input rows of surviving row groups (groups 1+2)");
        }
      } finally {
        closeAll(filterCols);
      }
    }
  }

  /** Verifies takeFilterRowMask() throws IllegalStateException after the reader is closed. */
  @Test
  void testTakeFilterRowMaskAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class, reader::takeFilterRowMask);
    }
  }

  // --------------------------------------------------------------------
  // Tests: setupChunkingForPayloadColumns()
  // --------------------------------------------------------------------

  /**
   * Verifies setupChunkingForPayloadColumns() activates payload-column chunking:
   * hasNextTableChunk() reports true after setup.
   */
  @Test
  void testSetupChunkingForPayloadColumnsActivatesPipeline(@TempDir Path tmp) throws IOException {
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
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          file, reader.filterColumnChunksByteRanges(survived));
      DeviceMemoryBuffer[] payloadCols = copyRangesToDevice(
          file, reader.payloadColumnChunksByteRanges(survived));
      try {
        reader.setupChunkingForFilterColumns(0L, 0L, survived,
            UseDataPageMask.NO, HybridScanReader.RowMaskKind.ALL_TRUE, filterCols);
        while (reader.hasNextTableChunk()) {
          reader.materializeFilterColumnsChunk().close();
        }
        try (ColumnVector rowMask = reader.takeFilterRowMask()) {
          reader.setupChunkingForPayloadColumns(0L, 0L, survived,
              rowMask, UseDataPageMask.NO, payloadCols);
          assertTrue(reader.hasNextTableChunk(), "payload chunking active after setup");
        }
      } finally {
        closeAll(filterCols);
        closeAll(payloadCols);
      }
    }
  }

  /** Verifies setupChunkingForPayloadColumns() throws IllegalArgumentException for a null row-group array. */
  @Test
  void testSetupChunkingForPayloadColumnsRejectsNullRowGroups(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null);
         ColumnVector dummyMask = ColumnVector.fromBooleans(true)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.setupChunkingForPayloadColumns(0L, 0L, null, dummyMask,
              UseDataPageMask.NO, new DeviceMemoryBuffer[0]));
    }
  }

  /** Verifies setupChunkingForPayloadColumns() throws IllegalArgumentException for a null row mask. */
  @Test
  void testSetupChunkingForPayloadColumnsRejectsNullRowMask(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.setupChunkingForPayloadColumns(0L, 0L, new int[]{0}, null,
              UseDataPageMask.NO, new DeviceMemoryBuffer[0]));
    }
  }

  /** Verifies setupChunkingForPayloadColumns() throws IllegalStateException after the reader is closed. */
  @Test
  void testSetupChunkingForPayloadColumnsAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         ColumnVector dummyMask = ColumnVector.fromBooleans(true)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class,
          () -> reader.setupChunkingForPayloadColumns(0L, 0L, new int[]{0}, dummyMask,
              UseDataPageMask.NO, new DeviceMemoryBuffer[0]));
    }
  }

  // --------------------------------------------------------------------
  // Tests: materializePayloadColumnsChunk()
  // --------------------------------------------------------------------

  /**
   * Verifies materializePayloadColumnsChunk() drains the exact expected row count when
   * chained after setupChunkingForPayloadColumns(): 1599 rows surviving the filter
   * zip_code &gt; 150,000.
   */
  @Test
  void testMaterializePayloadColumnsChunkExactTotal(@TempDir Path tmp) throws IOException {
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
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          file, reader.filterColumnChunksByteRanges(survived));
      DeviceMemoryBuffer[] payloadCols = copyRangesToDevice(
          file, reader.payloadColumnChunksByteRanges(survived));
      try {
        reader.setupChunkingForFilterColumns(0L, 0L, survived,
            UseDataPageMask.NO, HybridScanReader.RowMaskKind.ALL_TRUE, filterCols);
        while (reader.hasNextTableChunk()) {
          reader.materializeFilterColumnsChunk().close();
        }
        try (ColumnVector rowMask = reader.takeFilterRowMask()) {
          reader.setupChunkingForPayloadColumns(0L, 0L, survived,
              rowMask, UseDataPageMask.NO, payloadCols);
          long total = 0;
          while (reader.hasNextTableChunk()) {
            try (Table chunk = reader.materializePayloadColumnsChunk(rowMask)) {
              total += chunk.getRowCount();
            }
          }
          assertEquals(1599L, total);
        }
      } finally {
        closeAll(filterCols);
        closeAll(payloadCols);
      }
    }
  }

  /** Verifies materializePayloadColumnsChunk() throws IllegalArgumentException for a null row mask. */
  @Test
  void testMaterializePayloadColumnsChunkRejectsNullRowMask(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.materializePayloadColumnsChunk(null));
    }
  }

  /** Verifies materializePayloadColumnsChunk() throws IllegalStateException after the reader is closed. */
  @Test
  void testMaterializePayloadColumnsChunkAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         ColumnVector dummyMask = ColumnVector.fromBooleans(true)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class,
          () -> reader.materializePayloadColumnsChunk(dummyMask));
    }
  }

  // --------------------------------------------------------------------
  // Tests: setupChunkingForAllColumns() / materializeAllColumnsChunk()
  //
  // These two methods are tightly coupled (every meaningful scenario calls them in
  // sequence with hasNextTableChunk() driving the loop). The integration test below
  // exercises both together; per-method negatives are co-located.
  // --------------------------------------------------------------------

  /**
   * Verifies the all-columns chunked pipeline drains the exact total row count (3,000)
   * with all 3 projected columns in every chunk.
   */
  @Test
  void testChunkedAllColumnsExactTotal(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
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
        assertEquals(3000L, totalRows);
      } finally {
        closeAll(devs);
      }
    }
  }

  /** Verifies setupChunkingForAllColumns() throws IllegalArgumentException for a null row-group array. */
  @Test
  void testSetupChunkingForAllColumnsRejectsNullRowGroups(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.setupChunkingForAllColumns(0L, 0L, null, new DeviceMemoryBuffer[0]));
    }
  }

  /** Verifies setupChunkingForAllColumns() throws IllegalStateException after the reader is closed. */
  @Test
  void testSetupChunkingForAllColumnsAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class,
          () -> reader.setupChunkingForAllColumns(0L, 0L, new int[]{0}, new DeviceMemoryBuffer[0]));
    }
  }

  /** Verifies materializeAllColumnsChunk() throws IllegalStateException after the reader is closed. */
  @Test
  void testMaterializeAllColumnsChunkAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class, reader::materializeAllColumnsChunk);
    }
  }

  // --------------------------------------------------------------------
  // Tests: hasNextTableChunk()
  // --------------------------------------------------------------------

  /**
   * Verifies hasNextTableChunk() reports the active lifecycle of a chunked pipeline: true
   * after setup, then false once all chunks have been drained. hasNextTableChunk() is not a
   * soft probe — it requires an active chunking pipeline; calling it before any setup
   * raises a CudfException ("Chunking not yet setup") from the C++ implementation.
   */
  @Test
  void testHasNextTableChunkActiveAndAfterDrain(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      int[] rgs = reader.allRowGroups();
      DeviceMemoryBuffer[] devs = copyRangesToDevice(
          file, reader.allColumnChunksByteRanges(rgs));
      try {
        reader.setupChunkingForAllColumns(0L, 0L, rgs, devs);
        assertTrue(reader.hasNextTableChunk(), "chunking active after setup");
        while (reader.hasNextTableChunk()) {
          reader.materializeAllColumnsChunk().close();
        }
        assertFalse(reader.hasNextTableChunk(), "no further chunks after drain");
      } finally {
        closeAll(devs);
      }
    }
  }

  /** Verifies hasNextTableChunk() throws IllegalStateException after the reader is closed. */
  @Test
  void testHasNextTableChunkAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class, reader::hasNextTableChunk);
    }
  }

  // --------------------------------------------------------------------
  // Tests: constructRowGroupPasses()
  // --------------------------------------------------------------------

  /**
   * Verifies constructRowGroupPasses() with no read-limit (passReadLimit = 0) packs all
   * input row groups into a single pass that is structurally equal to the input.
   */
  @Test
  void testConstructRowGroupPassesUnlimitedReturnsSinglePass(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      int[] all = reader.allRowGroups();
      int[][] passes = reader.constructRowGroupPasses(all, 0L);
      assertEquals(1, passes.length, "passReadLimit = 0 must return one pass");
      assertArrayEquals(all, passes[0]);
    }
  }

  /**
   * Verifies that the union of all row groups across all returned passes equals the input
   * set: no row group is dropped or duplicated regardless of how passes are partitioned.
   */
  @Test
  void testConstructRowGroupPassesUnionEqualsInput(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      int[] all = reader.allRowGroups();
      int[][] passes = reader.constructRowGroupPasses(all, 0L);
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

  /** Verifies constructRowGroupPasses() throws IllegalArgumentException for a null row-group array. */
  @Test
  void testConstructRowGroupPassesRejectsNull(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip_code", "num_units"), null)) {
      assertThrows(IllegalArgumentException.class,
          () -> reader.constructRowGroupPasses(null, 0L));
    }
  }

  /** Verifies constructRowGroupPasses() throws IllegalStateException after the reader is closed. */
  @Test
  void testConstructRowGroupPassesAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip_code", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class,
          () -> reader.constructRowGroupPasses(new int[]{0}, 0L));
    }
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
   * Writes a small Parquet file with {@code ROWGROUP}-level statistics: row-group min/max
   * are recorded but no page index (no {@code ColumnIndex}/{@code OffsetIndex}) is emitted.
   * Used as the negative case for {@link #testPageIndexByteRangeEmptyForRowGroupStats}.
   */
  private static void writeRowGroupStatsParquet(File path) {
    int rows = 100;
    ParquetWriterOptions opts = ParquetWriterOptions.builder()
        .withNonNullableColumns("id", "zip_code")
        .withRowGroupSizeRows(rows)
        .withStatisticsFrequency(ParquetWriterOptions.StatisticsFrequency.ROWGROUP)
        .build();
    try (TableWriter writer = Table.writeParquetChunked(opts, path);
         ColumnVector id = ColumnVector.fromInts(IntStream.range(0, rows).toArray());
         ColumnVector zipCode = ColumnVector.fromInts(
             IntStream.range(0, rows).map(i -> 10000 + i).toArray());
         Table t = new Table(id, zipCode)) {
      writer.write(t);
    }
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
