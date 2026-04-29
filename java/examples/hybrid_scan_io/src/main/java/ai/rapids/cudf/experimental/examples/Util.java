/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.experimental.examples;

import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.experimental.ByteRange;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;

/**
 * Shared helpers for the hybrid-scan Java examples. Mirrors the lightweight pieces of
 * {@code cpp/examples/hybrid_scan_io/io_utils.hpp}.
 */
public final class Util {
  private Util() {}

  /** Magic + length suffix at the end of every Parquet file. */
  private static final int FOOTER_TAIL_BYTES = 8;

  /**
   * Copy the entire file contents into a {@link HostMemoryBuffer}.
   */
  public static HostMemoryBuffer readFileToHostBuffer(File file) throws IOException {
    byte[] bytes = Files.readAllBytes(file.toPath());
    HostMemoryBuffer buf = HostMemoryBuffer.allocate(bytes.length);
    buf.setBytes(0, bytes, 0, bytes.length);
    return buf;
  }

  /**
   * Slice the Parquet footer out of a host buffer that holds the full file.
   */
  public static HostMemoryBuffer extractFooter(HostMemoryBuffer file) {
    long fileLen = file.getLength();
    int footerLen = file.getInt(fileLen - FOOTER_TAIL_BYTES);
    long footerStart = fileLen - FOOTER_TAIL_BYTES - footerLen;
    return file.slice(footerStart, footerLen);
  }

  /**
   * Read just the last {@code suffixLen + footerLen} bytes from a file, given the suffix length.
   * For most files {@code suffixLen} is 8 (the 4-byte footer length + 4-byte magic).
   *
   * <p>This avoids copying the entire file into memory when only the footer is needed.
   */
  public static HostMemoryBuffer readFooterOnly(File file) throws IOException {
    try (RandomAccessFile raf = new RandomAccessFile(file, "r");
         FileChannel ch = raf.getChannel()) {
      long fileLen = ch.size();
      // Read the last 8 bytes (footer length + magic).
      ByteBuffer tail = ByteBuffer.allocate(FOOTER_TAIL_BYTES).order(ByteOrder.LITTLE_ENDIAN);
      ch.read(tail, fileLen - FOOTER_TAIL_BYTES);
      tail.flip();
      int footerLen = tail.getInt();
      // Now read the actual footer bytes.
      HostMemoryBuffer footer = HostMemoryBuffer.allocate(footerLen);
      try {
        byte[] bytes = new byte[footerLen];
        ByteBuffer footerBb = ByteBuffer.wrap(bytes);
        long footerStart = fileLen - FOOTER_TAIL_BYTES - footerLen;
        long pos = footerStart;
        while (footerBb.hasRemaining()) {
          int n = ch.read(footerBb, pos);
          if (n < 0) {
            throw new IOException("Unexpected EOF while reading parquet footer");
          }
          pos += n;
        }
        footer.setBytes(0, bytes, 0, bytes.length);
        return footer;
      } catch (IOException | RuntimeException e) {
        footer.close();
        throw e;
      }
    }
  }

  /**
   * Copy each {@link ByteRange} from a host buffer into its own {@link DeviceMemoryBuffer}.
   * Caller owns the returned buffers and must close them.
   */
  public static DeviceMemoryBuffer[] copyRangesToDevice(HostMemoryBuffer file,
                                                        ByteRange[] ranges) {
    DeviceMemoryBuffer[] out = new DeviceMemoryBuffer[ranges.length];
    try {
      for (int i = 0; i < ranges.length; i++) {
        ByteRange r = ranges[i];
        DeviceMemoryBuffer dev = DeviceMemoryBuffer.allocate(r.size());
        try (HostMemoryBuffer slice = file.slice(r.offset(), r.size())) {
          dev.copyFromHostBuffer(slice);
        }
        out[i] = dev;
      }
      return out;
    } catch (Throwable t) {
      closeAll(out);
      throw t;
    }
  }

  /** Best-effort close-all for an array of {@link DeviceMemoryBuffer}; ignores nulls. */
  public static void closeAll(DeviceMemoryBuffer[] buffers) {
    if (buffers == null) return;
    for (DeviceMemoryBuffer b : buffers) {
      if (b != null) b.close();
    }
  }
}
