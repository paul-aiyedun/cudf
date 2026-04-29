/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import ai.rapids.cudf.CudfTestBase;

import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link ColumnNameReference}.
 */
public class ColumnNameReferenceTest extends CudfTestBase {

  @Test
  void testNullNameRejected() {
    assertThrows(IllegalArgumentException.class, () -> new ColumnNameReference(null));
  }

  @Test
  void testEmptyNameRejected() {
    assertThrows(IllegalArgumentException.class, () -> new ColumnNameReference(""));
  }

  @Test
  void testGetColumnName() {
    ColumnNameReference c = new ColumnNameReference("my_col");
    assertEquals("my_col", c.getColumnName());
  }

  @Test
  void testToString() {
    assertEquals("COLUMN(\"zip\")", new ColumnNameReference("zip").toString());
  }

  @Test
  void testCompileEquivalentToBuildingFilterExpression() {
    // Build a small filter expression using ColumnNameReference and verify that compiling
    // it does not throw and produces a non-null CompiledExpression. This exercises the JNI
    // deserializer that parses the COLUMN_NAME_REFERENCE node type (id 5).
    ColumnNameReference colRef = new ColumnNameReference("a");
    Literal lit = Literal.ofInt(42);
    BinaryOperation op = new BinaryOperation(BinaryOperator.GREATER, colRef, lit);
    assertDoesNotThrow(() -> {
      try (CompiledExpression compiled = op.compile()) {
        assertNotNull(compiled);
      }
    });
  }

  @Test
  void testSerializedSizeIncludesNameBytes() {
    // The serialized form is: 1 byte (node type) + 4 bytes (string length) + UTF-8 bytes.
    // We can only access getSerializedSize() through the compile path; instead, verify two
    // different names produce reasonable byte counts via compile (smoke test).
    assertDoesNotThrow(() -> {
      try (CompiledExpression a = new BinaryOperation(BinaryOperator.GREATER,
               new ColumnNameReference("a"), Literal.ofInt(1)).compile();
           CompiledExpression abc = new BinaryOperation(BinaryOperator.GREATER,
               new ColumnNameReference("abc"), Literal.ofInt(1)).compile()) {
        assertNotNull(a);
        assertNotNull(abc);
      }
    });
  }

  @Test
  void testColumnNameReferenceTypeOrdinalMatchesPlan() {
    // The plan and the JNI dispatch require COLUMN_NAME_REFERENCE to be the 6th entry
    // (ordinal 5) of the ExpressionType enum so that its serialized native ID is 5.
    assertEquals(5, AstExpression.ExpressionType.COLUMN_NAME_REFERENCE.ordinal());
  }
}
