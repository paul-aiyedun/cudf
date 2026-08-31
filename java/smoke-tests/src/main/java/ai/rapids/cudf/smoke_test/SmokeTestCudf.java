/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf.smoke_test;

import ai.rapids.cudf.BinaryOp;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.DefaultHostMemoryAllocator;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostBufferConsumer;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.MultiBufferDataSource;
import ai.rapids.cudf.ParquetChunkedReader;
import ai.rapids.cudf.ParquetOptions;
import ai.rapids.cudf.ParquetWriterOptions;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.TableWriter;
import ai.rapids.cudf.ast.ColumnReference;
import ai.rapids.cudf.ast.CompiledExpression;
import ai.rapids.cudf.nvcomp.BatchedLZ4Compressor;
import ai.rapids.cudf.nvcomp.BatchedLZ4Decompressor;

/**
 * Smoke test for a published cudf-java classifier JAR. Runs once per JAR with
 * only {@code ai.rapids:cudf} + slf4j on the classpath to verify that the
 * packaged native libraries load and a handful of core APIs work end-to-end.
 */
public final class SmokeTestCudf {
  private SmokeTestCudf() {}

  private static int stepCounter = 0;

  private static void runStep(String label, Runnable body) {
    stepCounter++;
    System.out.printf("[%d] %s ...%n", stepCounter, label);
    body.run();
    System.out.println("OK: " + label);
  }

  private static void check(boolean cond, String msg) {
    if (!cond) {
      throw new IllegalStateException("ASSERT: " + msg);
    }
  }

  /** Collect parquet bytes written via {@link HostBufferConsumer}. */
  private static final class CollectingConsumer implements HostBufferConsumer, AutoCloseable {
    private final HostMemoryBuffer buffer = HostMemoryBuffer.allocate(1024 * 1024);
    private long offset = 0;

    @Override
    public void handleBuffer(HostMemoryBuffer src, long len) {
      try {
        buffer.copyFromHostBuffer(offset, src, 0, len);
        offset += len;
      } finally {
        src.close();
      }
    }

    long length() {
      return offset;
    }

    HostMemoryBuffer buffer() {
      return buffer;
    }

    @Override
    public void close() {
      buffer.close();
    }
  }

  /**
   * Parquet write/read round-trip via {@link ParquetChunkedReader}'s DataSource
   * ctor. That path also emits {@code CUDF_LOG_WARN}, covering the linked
   * rapids_logger / spdlog stack.
   */
  private static void parquetChunkedLoggerSmokeTest() {
    ParquetWriterOptions writeOpts = ParquetWriterOptions.builder().withColumns(false, "a").build();
    try (CollectingConsumer consumer = new CollectingConsumer()) {
      try (ColumnVector col = ColumnVector.fromInts(1, 2, 3, 4, 5);
           Table table = new Table(col);
           TableWriter writer = Table.writeParquetChunked(writeOpts, consumer)) {
        writer.write(table);
      }
      check(consumer.length() > 0, "expected non-empty parquet bytes");
      // Non-zero chunk limit: DataSource reader derives pass limit and CUDF_LOG_WARN.
      final long chunkReadLimit = 64 * 1024L;
      try (HostMemoryBuffer slice = consumer.buffer().slice(0, consumer.length());
           MultiBufferDataSource ds = new MultiBufferDataSource(slice);
           ParquetChunkedReader reader =
               new ParquetChunkedReader(chunkReadLimit, ParquetOptions.DEFAULT, ds)) {
        long rows = 0;
        while (reader.hasNext()) {
          try (Table chunk = reader.readChunk()) {
            if (chunk != null) {
              rows += chunk.getRowCount();
            }
          }
        }
        check(rows == 5, "expected 5 rows from chunked parquet logger smoke test");
      }
    }
  }

  /**
   * LZ4 compress/decompress round-trip via {@code ai.rapids.cudf.nvcomp},
   * covering nvcomp symbols linked into libcudf.
   */
  private static void nvcompLz4RoundTrip() {
    final long chunkSize = 64 * 1024;
    final Cuda.Stream stream = Cuda.DEFAULT_STREAM;
    final long[] data = {0, 1, 2, 3, 4, 5, 6, 7};

    try (HostMemoryBuffer hostIn = DefaultHostMemoryAllocator.get().allocate(data.length * 8L);
         DeviceMemoryBuffer original = DeviceMemoryBuffer.allocate(hostIn.getLength());
         DeviceMemoryBuffer decompressed = DeviceMemoryBuffer.allocate(hostIn.getLength())) {
      hostIn.setLongs(0, data, 0, data.length);
      original.copyFromHostBuffer(hostIn);
      original.incRefCount(); // compress() closes its inputs

      DeviceMemoryBuffer[] compressed =
          new BatchedLZ4Compressor(chunkSize, Long.MAX_VALUE)
              .compress(new DeviceMemoryBuffer[]{original}, stream);
      check(compressed.length == 1 && compressed[0].getLength() > 0,
          "expected one non-empty compressed buffer");

      new BatchedLZ4Decompressor(chunkSize)
          .decompressAsync(compressed, new DeviceMemoryBuffer[]{decompressed}, stream);
      stream.sync();

      try (HostMemoryBuffer hostOut =
               DefaultHostMemoryAllocator.get().allocate(decompressed.getLength())) {
        hostOut.copyFromDeviceBuffer(decompressed);
        check(hostOut.getLength() == hostIn.getLength(), "decompressed size mismatch");
        for (int i = 0; i < data.length; i++) {
          check(hostOut.getLong(i * 8L) == data[i], "nvcomp mismatch at long[" + i + "]");
        }
      }
    }
  }

  public static void main(String[] args) {
    try (ColumnVector ints = ColumnVector.fromInts(1, 2, 3, 4, 5)) {
      runStep("Native deps load", () ->
          check(ints.getRowCount() == 5, "expected 5 rows after fromInts"));

      runStep("ColumnVector + Table", () -> {
        try (ColumnVector more = ColumnVector.fromInts(10, 20, 30, 40, 50);
             Table table = new Table(ints, more)) {
          check(table.getNumberOfColumns() == 2, "expected 2 columns");
          check(table.getRowCount() == 5, "expected 5 table rows");
        }
      });

      runStep("Filter", () -> {
        try (Scalar three = Scalar.fromInt(3);
             ColumnVector mask = ints.binaryOp(BinaryOp.GREATER, three, DType.BOOL8);
             Table table = new Table(ints);
             Table filtered = table.filter(mask)) {
          check(filtered.getRowCount() == 2, "expected 2 rows after filter (>3)");
        }
      });

      runStep("Aggregation (sum)", () -> {
        try (Scalar sum = ints.sum(DType.INT64)) {
          check(sum.isValid(), "sum scalar should be valid");
          check(sum.getLong() == 15L, "sum should be 15");
        }
      });

      runStep("String column create + length", () -> {
        try (ColumnVector strs = ColumnVector.fromStrings("a", "bb", "ccc");
             ColumnVector lengths = strs.getCharLengths();
             HostColumnVector hostLens = lengths.copyToHost()) {
          check(strs.getRowCount() == 3, "expected 3 string rows");
          check(hostLens.getInt(0) == 1, "len[0]==1");
          check(hostLens.getInt(1) == 2, "len[1]==2");
          check(hostLens.getInt(2) == 3, "len[2]==3");
        }
      });

      runStep("Host round-trip", () -> {
        try (HostColumnVector host = ints.copyToHost()) {
          check(host.getInt(0) == 1, "host[0]==1");
          check(host.getInt(4) == 5, "host[4]==5");
        }
      });

      runStep("AST computeColumn", () -> {
        try (Table table = new Table(ints);
             CompiledExpression expression = new ColumnReference(0).compile();
             ColumnVector result = expression.computeColumn(table);
             HostColumnVector host = result.copyToHost()) {
          check(result.getRowCount() == 5, "expected 5 AST result rows");
          check(host.getInt(0) == 1, "AST result[0]==1");
          check(host.getInt(4) == 5, "AST result[4]==5");
        }
      });

      runStep("nvcomp LZ4 round-trip", SmokeTestCudf::nvcompLz4RoundTrip);
      runStep("Parquet chunked logger smoke test", SmokeTestCudf::parquetChunkedLoggerSmokeTest);
    }
    System.out.println("ALL STEPS PASSED");
  }
}
