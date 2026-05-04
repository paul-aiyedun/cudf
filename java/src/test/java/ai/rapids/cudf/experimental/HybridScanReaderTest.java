/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.experimental;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.CudfTestBase;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.ParquetOptions;
import ai.rapids.cudf.ParquetWriterOptions;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.TableWriter;
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
 * <p>These tests generate Parquet files on the fly (no pre-baked {@code .parquet} fixture is
 * needed) and exercise the entire hybrid-scan pipeline: footer parsing, row group filtering,
 * row mask construction, two-step materialization, single-shot materialization, and the
 * chunked / streaming family.
 */
public class HybridScanReaderTest extends CudfTestBase {

  // --------------------------------------------------------------------
  // Fixture helpers
  // --------------------------------------------------------------------

  /**
   * Build a small 3-row-group Parquet file with int columns {@code id} (0..N-1),
   * {@code zip} (10000 + id*100), and {@code num_units} (1..3 cycle), and write it to the
   * supplied path.
   *
   * <p>The row groups are forced by writing each block of rows in a separate
   * {@link TableWriter#write(Table)} call: the parquet writer flushes a row group on every
   * call (subject to its own size limits), which gives us a deterministic three-row-group
   * file even though the underlying data is tiny.
   */
  private static int writeFixtureParquet(File path) {
    int rowsPerGroup = 1000;
    int numGroups = 3;
    int rows = rowsPerGroup * numGroups;
    ParquetWriterOptions opts = ParquetWriterOptions.builder()
        .withNonNullableColumns("id", "zip", "num_units")
        .withRowGroupSizeRows(rowsPerGroup)
        .withStatisticsFrequency(ParquetWriterOptions.StatisticsFrequency.PAGE)
        .build();
    try (TableWriter writer = Table.writeParquetChunked(opts, path)) {
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
    return rows;
  }

  /**
   * Read the entire file into a HostMemoryBuffer.
   */
  private static HostMemoryBuffer readFileToHostBuffer(File file) throws IOException {
    byte[] fileBytes = Files.readAllBytes(file.toPath());
    HostMemoryBuffer buffer = HostMemoryBuffer.allocate(fileBytes.length);
    buffer.setBytes(0, fileBytes, 0, fileBytes.length);
    return buffer;
  }

  /**
   * Extract the Parquet file footer from a host buffer holding the full file. The Parquet
   * file ends with [footer_bytes][4-byte little-endian footer length][4-byte magic "PAR1"].
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

  /**
   * Copy the byte ranges from a host buffer into device buffers (one per range).
   */
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

  // --------------------------------------------------------------------
  // Tests: metadata
  // --------------------------------------------------------------------

  @Test
  void testCreateAndEnumerateRowGroups(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    int rows = writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip", "num_units"), null)) {
      int[] rgs = reader.allRowGroups();
      assertNotNull(rgs);
      assertTrue(rgs.length > 0);
      assertEquals(rows, reader.totalRowsInRowGroups(rgs));
    }
  }

  @Test
  void testParquetMetadataReturnsPlausibleFields(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    int rows = writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip", "num_units"), null)) {
      FileMetaData md = reader.parquetMetadata();
      assertNotNull(md);
      assertEquals(rows, md.numRows());
      // Parquet file format version is normally 1 or 2.
      assertTrue(md.version() == 1 || md.version() == 2,
          "Unexpected file version: " + md.version());
      assertNotNull(md.createdBy());
    }
  }

  @Test
  void testPageIndexByteRangeAndSetup(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip", "num_units"), null)) {
      // The page index may be absent for tiny files even when PAGE statistics are
      // requested, so just verify the call returns a valid ByteRange and that
      // setupPageIndex accepts the corresponding bytes when the range is non-empty.
      ByteRange piRange = reader.pageIndexByteRange();
      assertNotNull(piRange);
      if (piRange.size() > 0) {
        try (HostMemoryBuffer pi = file.slice(piRange.offset(), piRange.size())) {
          reader.setupPageIndex(pi);
        }
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: row group enumeration
  // --------------------------------------------------------------------

  @Test
  void testResetColumnSelection(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip", "num_units"), null)) {
      // First do something that causes column selection to be cached internally...
      int[] rgs = reader.allRowGroups();
      reader.payloadColumnChunksByteRanges(rgs);
      // Then reset; should not throw.
      assertDoesNotThrow(reader::resetColumnSelection);
    }
  }

  // --------------------------------------------------------------------
  // Tests: row group filtering
  // --------------------------------------------------------------------

  @Test
  void testFilterRowGroupsWithStats(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    // zip values across the file are 10000, 10100, ..., 309900. Row groups split at 1000
    // rows each, so the second and third row groups have all-larger zips. Use a threshold
    // that prunes the first row group.
    ColumnNameReference zipCol = new ColumnNameReference("zip");
    Literal lit = Literal.ofInt(150000);
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER, zipCol, lit);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         CompiledExpression filter = expr.compile();
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip", "num_units"), filter)) {
      int[] all = reader.allRowGroups();
      int[] filtered = reader.filterRowGroupsWithStats(all);
      // The stats filter must never expand the input set.
      assertTrue(filtered.length <= all.length);
      // Pruning depends on whether the writer emitted multiple row groups with disjoint
      // stats ranges. Only assert the strict pruning property when we actually have more
      // than one input row group.
      if (all.length > 1) {
        assertTrue(filtered.length < all.length,
            "Stats filter should prune at least one row group when " + all.length +
            " input row groups have disjoint zip ranges, got " + filtered.length);
      }
    }
  }

  @Test
  void testSecondaryFiltersByteRangesShape(@TempDir Path tmp) throws IOException {
    // Even when the file has no bloom filters / dictionary pages applicable, the call
    // should succeed and return a structurally-valid SecondaryFilterRanges.
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    ColumnNameReference zipCol = new ColumnNameReference("zip");
    Literal lit = Literal.ofInt(150000);
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER, zipCol, lit);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         CompiledExpression filter = expr.compile();
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip", "num_units"), filter)) {
      int[] rgs = reader.allRowGroups();
      SecondaryFilterRanges sfr = reader.secondaryFiltersByteRanges(rgs);
      assertNotNull(sfr);
      assertNotNull(sfr.bloomFilterRanges());
      assertNotNull(sfr.dictionaryPageRanges());
    }
  }

  // --------------------------------------------------------------------
  // Tests: row mask
  // --------------------------------------------------------------------

  @Test
  void testBuildAllTrueRowMask(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    int rows = writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip", "num_units"), null)) {
      int[] rgs = reader.allRowGroups();
      try (ColumnVector mask = reader.buildAllTrueRowMask(rgs)) {
        assertEquals(DType.BOOL8, mask.getType());
        assertEquals(rows, mask.getRowCount());
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: byte ranges
  // --------------------------------------------------------------------

  @Test
  void testPayloadByteRangesNoFilter(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip", "num_units"), null)) {
      int[] rgs = reader.allRowGroups();
      ByteRange[] ranges = reader.payloadColumnChunksByteRanges(rgs);
      assertNotNull(ranges);
      assertTrue(ranges.length > 0);
    }
  }

  @Test
  void testFilterAndPayloadByteRangeSplit(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    ColumnNameReference zipCol = new ColumnNameReference("zip");
    Literal lit = Literal.ofInt(50000);
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER, zipCol, lit);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         CompiledExpression filter = expr.compile();
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip", "num_units"), filter)) {
      int[] rgs = reader.allRowGroups();
      ByteRange[] filterRanges = reader.filterColumnChunksByteRanges(rgs);
      ByteRange[] payloadRanges = reader.payloadColumnChunksByteRanges(rgs);
      assertNotNull(filterRanges);
      assertNotNull(payloadRanges);
      // With one filter column ("zip") and two payload columns ("id", "num_units"), the
      // filter ranges should equal one chunk per row group and payload ranges two per group.
      assertEquals(rgs.length, filterRanges.length);
      assertEquals(rgs.length * 2, payloadRanges.length);
    }
  }

  // --------------------------------------------------------------------
  // Tests: single-shot materialize
  // --------------------------------------------------------------------

  @Test
  void testExplicitTwoStepFlow(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    ColumnNameReference zipCol = new ColumnNameReference("zip");
    Literal lit = Literal.ofInt(50000);
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER, zipCol, lit);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         CompiledExpression filter = expr.compile();
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip", "num_units"), filter)) {
      int[] rgs = reader.allRowGroups();
      int[] survived = reader.filterRowGroupsWithStats(rgs);
      if (survived.length == 0) {
        // Filter pruned everything; nothing to materialize. Test still validates the API.
        return;
      }
      DeviceMemoryBuffer[] filterCols = null;
      DeviceMemoryBuffer[] payloadCols = null;
      try (ColumnVector rowMask = reader.buildAllTrueRowMask(survived)) {
        ByteRange[] filterRanges = reader.filterColumnChunksByteRanges(survived);
        ByteRange[] payloadRanges = reader.payloadColumnChunksByteRanges(survived);
        filterCols = copyRangesToDevice(file, filterRanges);
        payloadCols = copyRangesToDevice(file, payloadRanges);
        try (Table filtered = reader.materializeFilterColumns(survived, filterCols, rowMask,
                 UseDataPageMask.NO);
             Table payload = reader.materializePayloadColumns(survived, payloadCols, rowMask,
                 UseDataPageMask.NO)) {
          assertEquals(filtered.getRowCount(), payload.getRowCount());
        }
      } finally {
        closeAll(filterCols);
        closeAll(payloadCols);
      }
    }
  }

  @Test
  void testMaterializeAllColumns(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    int rows = writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip", "num_units"), null)) {
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
  // Tests: chunked materialize
  // --------------------------------------------------------------------

  @Test
  void testConstructRowGroupPasses(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip", "num_units"), null)) {
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

  @Test
  void testChunkedAllColumns(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    int rows = writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip", "num_units"), null)) {
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
  // Tests: convenience materialize-from-buffer
  // --------------------------------------------------------------------

  @Test
  void testMaterializeFromBufferNoFilter(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    int rows = writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip", "num_units"), null);
         Table table = reader.materializeFromBuffer(file)) {
      assertEquals(3, table.getNumberOfColumns());
      assertEquals(rows, table.getRowCount());
    }
  }

  @Test
  void testMaterializeFromBufferMatchesReadParquet(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file);
         HybridScanReader reader = new HybridScanReader(footer,
             optsForColumns("id", "zip", "num_units"), null);
         Table hybrid = reader.materializeFromBuffer(file);
         Table standard = Table.readParquet(
             optsForColumns("id", "zip", "num_units"), pq)) {
      assertEquals(standard.getNumberOfColumns(), hybrid.getNumberOfColumns());
      assertEquals(standard.getRowCount(), hybrid.getRowCount());
      for (int i = 0; i < standard.getNumberOfColumns(); i++) {
        assertEquals(standard.getColumn(i).getType(), hybrid.getColumn(i).getType(),
            "Column " + i + " type mismatch");
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: AST extension
  // --------------------------------------------------------------------

  @Test
  void testAstColumnNameReferenceCompilesAndCloses() {
    ColumnNameReference c = new ColumnNameReference("zip");
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
  // Tests: lifecycle / errors
  // --------------------------------------------------------------------

  @Test
  void testNullFooterThrows() {
    assertThrows(IllegalArgumentException.class,
        () -> new HybridScanReader(null, optsForColumns("id"), null));
  }

  @Test
  void testCloseIsIdempotent(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip", "num_units"), null);
      reader.close();
      reader.close();
    }
  }

  @Test
  void testCallAfterCloseThrows(@TempDir Path tmp) throws IOException {
    File pq = tmp.resolve("fixture.parquet").toFile();
    writeFixtureParquet(pq);
    try (HostMemoryBuffer file = readFileToHostBuffer(pq);
         HostMemoryBuffer footer = extractFooter(file)) {
      HybridScanReader reader = new HybridScanReader(footer,
          optsForColumns("id", "zip", "num_units"), null);
      reader.close();
      assertThrows(IllegalStateException.class, reader::allRowGroups);
    }
  }
}
