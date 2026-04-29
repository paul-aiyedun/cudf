# cudf Java hybrid_scan_io examples

Java equivalents of the C++ examples under `cpp/examples/hybrid_scan_io/`. They demonstrate
how to use the experimental `ai.rapids.cudf.experimental.HybridScanReader` (the Java binding
for `cudf::io::parquet::experimental::hybrid_scan_reader`) to read Parquet files subject to
selective filter expressions.

## Examples

| Class | Mirrors | Purpose |
|---|---|---|
| `HybridScanIoExample` | `hybrid_scan_io.cpp` | Reads a file once with the legacy reader and once with the hybrid scan reader (two-step mode), printing the row counts and elapsed time. Useful as a smoke test. |
| `HybridScanPipelineExample` | `hybrid_scan_pipeline.cpp` | Splits the file into row-group passes, reads each pass as a stream of chunks via the chunked all-columns API. |

`Util.java` contains shared helpers: reading a Parquet file into a `HostMemoryBuffer`,
extracting the footer suffix, and copying byte ranges to device buffers.

## Building

The examples expect that the parent `ai.rapids:cudf` artifact has been built and installed
locally. From the repo root:

```bash
cd java
mvn -DskipTests install
cd examples/hybrid_scan_io
mvn package
```

## Running

After `mvn install` has been run on the parent project:

```bash
# Single/two-step example
mvn exec:java \
  -Dexec.mainClass=ai.rapids.cudf.experimental.examples.HybridScanIoExample \
  -Dexec.args="/path/to/file.parquet zip 90000"

# Chunked pipeline example
mvn exec:java \
  -Dexec.mainClass=ai.rapids.cudf.experimental.examples.HybridScanPipelineExample \
  -Dexec.args="/path/to/file.parquet 67108864 16777216"
```

The two trailing numeric arguments to `HybridScanPipelineExample` are the per-pass and
per-chunk byte limits; pass `0` for "no limit".

## Notes on filters

Because the Parquet schema may not be known to the application at AST construction time,
filter expressions for the hybrid scan reader use
`ai.rapids.cudf.ast.ColumnNameReference` to refer to columns by name (see the example
source). The reader resolves the names against the file schema during materialization.

## S3 / remote IO

The C++ examples include an `io_source` abstraction that supports HOST_BUFFER, PINNED_BUFFER
and DEVICE_BUFFER, and the C++ build optionally enables KvikIO-based S3 reads. The Java
binding does not currently expose remote IO, and the Java build is configured without KvikIO
S3 support. To benchmark against an S3-hosted file, download it locally and pass the local
path to these examples.
