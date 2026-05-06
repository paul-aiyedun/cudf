/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Pair of byte-range vectors returned by
 * {@link HybridScanReader#secondaryFiltersByteRanges(int[])}.
 *
 * <p>The two vectors describe, for the input row groups:
 * <ul>
 *   <li>{@link #bloomFilterRanges()} — file byte ranges containing per-column-chunk
 *       Parquet bloom filter blobs that can be used for row-group pruning of equality
 *       predicates;</li>
 *   <li>{@link #dictionaryPageRanges()} — file byte ranges of the column-chunk
 *       dictionary pages used for row-group pruning of (in)equality predicates.</li>
 * </ul>
 *
 * <p>Both lists may be empty. The ordering follows the C++ reader's ordering and is
 * meaningful: the i-th entry corresponds to the i-th column-chunk needing the
 * respective filter.
 *
 * <p>Mirrors the {@code std::pair<std::vector<byte_range_info>, std::vector<byte_range_info>>}
 * returned by {@code hybrid_scan_reader::secondary_filters_byte_ranges}.
 *
 * <p>The APIs in this file are experimental and subject to change.
 */
@Experimental
public final class SecondaryFilterRanges {
  private final ByteRange[] bloomFilterRanges;
  private final ByteRange[] dictionaryPageRanges;

  public SecondaryFilterRanges(ByteRange[] bloomFilterRanges,
                               ByteRange[] dictionaryPageRanges) {
    this.bloomFilterRanges =
        bloomFilterRanges == null ? new ByteRange[0] : bloomFilterRanges.clone();
    this.dictionaryPageRanges =
        dictionaryPageRanges == null ? new ByteRange[0] : dictionaryPageRanges.clone();
  }

  /** @return the bloom-filter byte ranges as an unmodifiable list. */
  public List<ByteRange> bloomFilterRanges() {
    return Arrays.asList(bloomFilterRanges);
  }

  /** @return the dictionary-page byte ranges as an unmodifiable list. */
  public List<ByteRange> dictionaryPageRanges() {
    return Arrays.asList(dictionaryPageRanges);
  }

  /** @return a defensive copy of the bloom-filter ranges. */
  public ByteRange[] bloomFilterRangesArray() {
    return bloomFilterRanges.clone();
  }

  /** @return a defensive copy of the dictionary-page ranges. */
  public ByteRange[] dictionaryPageRangesArray() {
    return dictionaryPageRanges.clone();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof SecondaryFilterRanges)) return false;
    SecondaryFilterRanges other = (SecondaryFilterRanges) o;
    return Arrays.equals(bloomFilterRanges, other.bloomFilterRanges) &&
           Arrays.equals(dictionaryPageRanges, other.dictionaryPageRanges);
  }

  @Override
  public int hashCode() {
    return Objects.hash(Arrays.hashCode(bloomFilterRanges),
                        Arrays.hashCode(dictionaryPageRanges));
  }

  @Override
  public String toString() {
    return "SecondaryFilterRanges{bloom=" + bloomFilterRanges.length +
           " ranges, dict=" + dictionaryPageRanges.length + " ranges}";
  }
}
