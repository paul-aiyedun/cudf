/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf.examples.joincache;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ContiguousTable;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.GatherMap;
import ai.rapids.cudf.HashJoin;
import ai.rapids.cudf.JoinMatchContext;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.Table;

import java.util.Arrays;
import java.util.function.LongSupplier;

/**
 * Times partitioned inner join retrieval with and without a {@link JoinMatchContext}.
 *
 * <p>The data is a star-schema join. A dimension table with unique keys is the build side and a
 * larger fact table with skewed foreign keys is the probe side.</p>
 */
public final class JoinMatchContextBenchmark {
  private JoinMatchContextBenchmark() {}

  public static void main(String[] args) {
    if (args.length != 4) {
      System.err.println("usage: JoinMatchContextBenchmark buildRows probeRows partitions iterations");
      System.exit(2);
    }
    int buildRows = Integer.parseInt(args[0]);
    int probeRows = Integer.parseInt(args[1]);
    int partitions = Integer.parseInt(args[2]);
    int iterations = Integer.parseInt(args[3]);

    Rmm.initialize(RmmAllocationMode.CUDA_ASYNC, null, Cuda.memGetInfo().free * 9 / 10);

    int[] bounds = new int[partitions + 1];
    for (int i = 0; i <= partitions; i++) {
      bounds[i] = (int) ((long) probeRows * i / partitions);
    }

    try (Scalar zero = Scalar.fromInt(0);
         ColumnVector dimKeys = ColumnVector.sequence(zero, buildRows);
         ColumnVector factKeys = skewedKeys(probeRows, buildRows);
         Table dim = new Table(dimKeys);
         Table fact = new Table(factKeys);
         HashJoin hashJoin = new HashJoin(dim, false)) {
      ContiguousTable[] slices = fact.contiguousSplit(Arrays.copyOfRange(bounds, 1, partitions));
      try {
        Result uncached = measure(iterations, () -> {
          long rows = 0;
          for (ContiguousTable slice : slices) {
            rows += countAndClose(slice.getTable().innerJoinGatherMaps(hashJoin));
          }
          return rows;
        });
        Result cached = measure(iterations, () -> {
          long rows = 0;
          try (JoinMatchContext ctx = fact.innerJoinMatchContext(hashJoin)) {
            for (int p = 0; p < partitions; p++) {
              rows += countAndClose(ctx.partitionGatherMaps(bounds[p], bounds[p + 1]));
            }
          }
          return rows;
        });

        System.out.printf("Star-schema inner join: dimension=%,d rows (unique keys), "
            + "fact=%,d rows (skewed keys), partitions=%d%n", buildRows, probeRows, partitions);
        System.out.printf("Fact rows matched: %.1f%%   fact rows hitting the top 1%% of keys: %.1f%%%n",
            100.0 * cached.rows / probeRows, 100.0 * hotRows(factKeys, buildRows) / probeRows);
        uncached.print("without cache");
        cached.print("with cache");
        System.out.printf("speedup=%.2fx%n", uncached.medianMs / cached.medianMs);
        if (uncached.rows != cached.rows) {
          System.err.printf("Row count mismatch: %d vs %d%n", uncached.rows, cached.rows);
          System.exit(1);
        }
      } finally {
        for (ContiguousTable slice : slices) {
          slice.close();
        }
      }
    }
  }

  // A fixed multiplicative hash of the row number gives a repeatable uniform u in [0, 1).
  // Raising u to the third power piles keys onto a few hot values. The key range is twice the
  // dimension size, so about one in five fact rows has no match.
  private static ColumnVector skewedKeys(int rows, int dimRows) {
    try (Scalar zero = Scalar.fromLong(0);
         Scalar multiplier = Scalar.fromLong(2654435761L);
         Scalar twoTo32 = Scalar.fromLong(1L << 32);
         Scalar skew = Scalar.fromDouble(3.0);
         Scalar range = Scalar.fromDouble(2.0 * dimRows);
         ColumnVector rowIds = ColumnVector.sequence(zero, rows);
         ColumnVector hashed = rowIds.mul(multiplier);
         ColumnVector wrapped = hashed.mod(twoTo32);
         ColumnVector uniform = wrapped.trueDiv(twoTo32, DType.FLOAT64);
         ColumnVector skewed = uniform.pow(skew);
         ColumnVector keys = skewed.mul(range)) {
      return keys.castTo(DType.INT32);
    }
  }

  private static long hotRows(ColumnVector keys, int dimRows) {
    try (Scalar limit = Scalar.fromInt(Math.max(1, dimRows / 100));
         ColumnVector isHot = keys.lessThan(limit);
         ColumnVector asInt = isHot.castTo(DType.INT64);
         Scalar total = asInt.sum()) {
      return total.getLong();
    }
  }

  private static long countAndClose(GatherMap[] maps) {
    long rows = maps[0].getRowCount();
    for (GatherMap map : maps) {
      map.close();
    }
    return rows;
  }

  private static Result measure(int iterations, LongSupplier run) {
    run.getAsLong();
    double[] ms = new double[iterations];
    long rows = 0;
    for (int i = 0; i < iterations; i++) {
      long start = System.nanoTime();
      rows = run.getAsLong();
      Cuda.DEFAULT_STREAM.sync();
      ms[i] = (System.nanoTime() - start) / 1e6;
    }
    Arrays.sort(ms);
    return new Result(ms[iterations / 2], ms[0], rows);
  }

  private static final class Result {
    final double medianMs;
    final double minMs;
    final long rows;

    Result(double medianMs, double minMs, long rows) {
      this.medianMs = medianMs;
      this.minMs = minMs;
      this.rows = rows;
    }

    void print(String label) {
      System.out.printf("%-14s median %9.2f ms   min %9.2f ms   output rows %,d%n",
          label + ":", medianMs, minMs, rows);
    }
  }
}
