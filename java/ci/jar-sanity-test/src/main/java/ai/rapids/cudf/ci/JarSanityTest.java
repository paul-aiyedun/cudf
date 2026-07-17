/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf.ci;

import ai.rapids.cudf.BinaryOp;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.ReductionAggregation;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.Table;

/**
 * Minimal GPU workload used to verify a gathered cuDF Java classifier JAR.
 *
 * <p>Each check prints {@code PASS: <name>} or {@code FAIL: <name>} so the
 * wrapper script can grep the output. The program exits 0 only when every
 * check passes.</p>
 */
public final class JarSanityTest {
  private static final long RMM_POOL_SIZE = 256L * 1024L * 1024L;

  private JarSanityTest() {}

  public static void main(String[] args) {
    int failures = 0;
    try {
      failures += check("cuda_env", checkCudaEnv());
      if (failures > 0) {
        System.exit(1);
      }

      Rmm.initialize(RmmAllocationMode.POOL, Rmm.logToStderr(), RMM_POOL_SIZE);
      try {
        failures += check("column_create", testColumnCreate());
        failures += check("binary_add", testBinaryAdd());
        failures += check("table_create", testTableCreate());
        failures += check("reduce_sum", testReduceSum());
      } finally {
        if (Rmm.isInitialized()) {
          Rmm.shutdown();
        }
      }
    } catch (Throwable t) {
      System.err.println("FAIL: unexpected_exception " + t);
      t.printStackTrace(System.err);
      System.exit(1);
    }

    if (failures > 0) {
      System.err.printf("JarSanityTest: %d check(s) failed%n", failures);
      System.exit(1);
    }
    System.out.println("JarSanityTest: all checks passed");
  }

  private static int check(String name, boolean ok) {
    if (ok) {
      System.out.println("PASS: " + name);
      return 0;
    }
    System.err.println("FAIL: " + name);
    return 1;
  }

  private static boolean checkCudaEnv() {
    if (!Cuda.isEnvCompatibleForTesting()) {
      System.err.println("No compatible CUDA GPU detected (Cuda.isEnvCompatibleForTesting=false)");
      return false;
    }
    return true;
  }

  private static boolean testColumnCreate() {
    try (ColumnVector cv = ColumnVector.fromInts(1, 2, 3, 4, 5)) {
      return cv.getRowCount() == 5L;
    }
  }

  private static boolean testBinaryAdd() {
    try (ColumnVector left = ColumnVector.fromInts(1, 2, 3);
         ColumnVector right = ColumnVector.fromInts(10, 20, 30);
         ColumnVector sum = left.binaryOp(BinaryOp.ADD, right, DType.INT32);
         HostColumnVector host = sum.copyToHost()) {
      if (host.getRowCount() != 3) {
        return false;
      }
      return host.getInt(0) == 11 && host.getInt(1) == 22 && host.getInt(2) == 33;
    }
  }

  private static boolean testTableCreate() {
    try (ColumnVector col = ColumnVector.fromInts(7, 8, 9);
         Table table = new Table(col)) {
      return table.getRowCount() == 3L && table.getNumberOfColumns() == 1;
    }
  }

  private static boolean testReduceSum() {
    try (ColumnVector col = ColumnVector.fromInts(1, 2, 3, 4);
         Scalar sum = col.reduce(ReductionAggregation.sum(), DType.INT32)) {
      return sum.getInt() == 10;
    }
  }
}
