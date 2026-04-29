/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.experimental.examples;

import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.ParquetOptions;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.experimental.ByteRange;
import ai.rapids.cudf.experimental.HybridScanReader;

import java.io.File;
import java.io.IOException;

/**
 * Java equivalent of {@code cpp/examples/hybrid_scan_io/hybrid_scan_pipeline.cpp}.
 *
 * <p>Splits a Parquet file into row-group passes and reads each pass as a stream of chunks
 * via the {@link HybridScanReader#materializeAllColumnsChunk()} API. Useful for very large
 * files where a single materialize call would not fit in memory.
 *
 * <p>Usage:
 * <pre>
 * mvn -pl java/examples/hybrid_scan_io exec:java \
 *     -Dexec.mainClass=ai.rapids.cudf.experimental.examples.HybridScanPipelineExample \
 *     -Dexec.args="/path/to/file.parquet [pass-bytes [chunk-bytes]]"
 * </pre>
 *
 * <p>Both byte limits default to {@code 0} (no limit).
 */
public final class HybridScanPipelineExample {
  private HybridScanPipelineExample() {}

  public static void main(String[] args) throws IOException {
    if (args.length < 1) {
      System.err.println(
          "Usage: HybridScanPipelineExample <parquet-file> [pass-bytes [chunk-bytes]]");
      System.exit(1);
    }
    File path = new File(args[0]);
    long passReadLimit = (args.length >= 2) ? Long.parseLong(args[1]) : 0L;
    long chunkReadLimit = (args.length >= 3) ? Long.parseLong(args[2]) : 0L;

    if (!path.isFile()) {
      System.err.println("Input file not found: " + path);
      System.exit(2);
    }

    try {
      if (!Rmm.isInitialized()) {
        Rmm.initialize(RmmAllocationMode.POOL, Rmm.logToStderr(), 512L * 1024L * 1024L);
      }

      try (HostMemoryBuffer file = Util.readFileToHostBuffer(path);
           HostMemoryBuffer footer = Util.extractFooter(file);
           HybridScanReader reader = new HybridScanReader(footer, ParquetOptions.DEFAULT, null)) {
        int[] allRowGroups = reader.allRowGroups();
        int[][] passes = reader.constructRowGroupPasses(allRowGroups, passReadLimit);
        System.out.printf("Split %d row groups into %d pass(es)%n",
            allRowGroups.length, passes.length);

        long totalRows = 0;
        long t0 = System.nanoTime();
        for (int p = 0; p < passes.length; p++) {
          int[] pass = passes[p];
          ByteRange[] ranges = reader.allColumnChunksByteRanges(pass);
          DeviceMemoryBuffer[] devs = Util.copyRangesToDevice(file, ranges);
          try {
            reader.setupChunkingForAllColumns(chunkReadLimit, passReadLimit, pass, devs);
            int chunks = 0;
            long passRows = 0;
            while (reader.hasNextTableChunk()) {
              try (Table chunk = reader.materializeAllColumnsChunk()) {
                passRows += chunk.getRowCount();
                chunks++;
              }
            }
            totalRows += passRows;
            System.out.printf("  pass %d: row_groups=%d chunks=%d rows=%d%n",
                p, pass.length, chunks, passRows);
          } finally {
            Util.closeAll(devs);
          }
        }
        long ms = (System.nanoTime() - t0) / 1_000_000L;
        System.out.printf("Pipeline read: total_rows=%d time_ms=%d%n", totalRows, ms);
      }
    } finally {
      if (Rmm.isInitialized()) {
        Rmm.shutdown();
      }
    }
  }
}
