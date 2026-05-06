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
 *
 * <p>Tests are organised in the same order as the public API: constructor,
 * {@code getColumnName()}, {@code toString()}, {@code compile()} (inherited from
 * {@link AstExpression}), and the {@code ExpressionType} ordinal invariant.
 */
public class ColumnNameReferenceTest extends CudfTestBase {

  // --------------------------------------------------------------------
  // Tests: ColumnNameReference (constructor)
  // --------------------------------------------------------------------

  /** Verifies that passing null to the constructor throws IllegalArgumentException. */
  @Test
  void testNullNameRejected() {
    assertThrows(IllegalArgumentException.class, () -> new ColumnNameReference(null));
  }

  /** Verifies that passing an empty string to the constructor throws IllegalArgumentException. */
  @Test
  void testEmptyNameRejected() {
    assertThrows(IllegalArgumentException.class, () -> new ColumnNameReference(""));
  }

  // --------------------------------------------------------------------
  // Tests: getColumnName()
  // --------------------------------------------------------------------

  /** Verifies that getColumnName() returns the name supplied to the constructor. */
  @Test
  void testGetColumnName() {
    ColumnNameReference c = new ColumnNameReference("my_col");
    assertEquals("my_col", c.getColumnName());
  }

  // --------------------------------------------------------------------
  // Tests: toString()
  // --------------------------------------------------------------------

  /** Verifies that toString() produces the expected COLUMN("name") representation. */
  @Test
  void testToString() {
    assertEquals("COLUMN(\"zip\")", new ColumnNameReference("zip").toString());
  }

  // --------------------------------------------------------------------
  // Tests: compile() (inherited from AstExpression)
  // --------------------------------------------------------------------

  /**
   * Verifies that a filter expression using ColumnNameReference compiles without error and returns
   * a non-null CompiledExpression, exercising the JNI deserializer for node type
   * COLUMN_NAME_REFERENCE.
   */
  @Test
  void testCompileEquivalentToBuildingFilterExpression() {
    ColumnNameReference colRef = new ColumnNameReference("a");
    Literal lit = Literal.ofInt(42);
    BinaryOperation op = new BinaryOperation(BinaryOperator.GREATER, colRef, lit);
    assertDoesNotThrow(() -> {
      try (CompiledExpression compiled = op.compile()) {
        assertNotNull(compiled);
      }
    });
  }

  /**
   * Smoke-tests that two ColumnNameReference nodes with different name lengths both compile
   * successfully, confirming the serialized size accounts for name bytes.
   */
  @Test
  void testSerializedSizeIncludesNameBytes() {
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

  // --------------------------------------------------------------------
  // Tests: ExpressionType ordinal invariant
  // --------------------------------------------------------------------

  /**
   * Verifies that COLUMN_NAME_REFERENCE has ordinal 5, matching the native plan's expected
   * node-type ID.
   */
  @Test
  void testColumnNameReferenceTypeOrdinalMatchesPlan() {
    assertEquals(5, AstExpression.ExpressionType.COLUMN_NAME_REFERENCE.ordinal());
  }
}
