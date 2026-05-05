# cudf Java hybrid_scan_io examples

Java equivalents of the C++ examples under `cpp/examples/hybrid_scan_io/`. They demonstrate
how to use the experimental `ai.rapids.cudf.experimental.HybridScanReader` (the Java binding
for `cudf::io::parquet::experimental::hybrid_scan_reader`) to read Parquet files subject to
selective filter expressions.

## Examples

| Class | Mirrors | Purpose |
|---|---|---|
| `GenerateSampleParquetFileMain` | — | Generates a sample three-row-group Parquet file used as input by the other examples. Run this first. |
| `HybridScanIoExample` | `hybrid_scan_io.cpp` | Reads a file once with the legacy reader and once with the hybrid scan reader (two-step mode), printing the row counts and elapsed time. Useful as a smoke test. |
| `HybridScanPipelineExample` | `hybrid_scan_pipeline.cpp` | Splits the file into row-group passes (batches), reads each pass as a stream of cuDF Table chunks via the chunked all-columns API. |

`Util.java` contains shared helpers: reading a Parquet file into a `HostMemoryBuffer`,
reading only the footer, and copying byte ranges to device buffers.

## Building

The examples expect that the parent `ai.rapids:cudf` artifact has been built and installed
locally. From the repo root:

```bash
cd java
mvn -DskipTests install   # installs cudf-java jar into ~/.m2 so examples can depend on it
cd examples/hybrid_scan_io
mvn package               # compiles and packages only this example module
```

## Generating sample data

The examples read from a Parquet file with three integer columns (`id`, `zip_code`,
`num_units`) split into three row groups of 1 000 rows each. Generate it before running
the examples:

```bash
# Generate the sample Parquet file used by the examples
mvn exec:java \
  -Dexec.mainClass=ai.rapids.cudf.examples.GenerateSampleParquetFileMain \
  -Dexec.args="/tmp/sample.parquet"
```

## Running

The fastest way to exercise all three examples is the bundled `run_examples.sh`:

```bash
./run_examples.sh                # data-gen + io + pipeline
./run_examples.sh -b             # also force `mvn package -DskipTests` first
./run_examples.sh --help         # full usage
```

`run_examples.sh`:

- Prechecks that the `ai.rapids:cudf` jar is installed in `~/.m2/repository`
  and prints a clear error (with the `mvn install -DskipTests` invocation)
  if it isn't.
- Auto-builds the example module on first run when
  `target/classes/ai/rapids/cudf/examples/` has no `*.class` files.
- Accepts `-b` / `--build` to force a rebuild even when classes are already
  present.

To run a single example by hand (after `mvn package` has been run on the
examples module):

```bash
# Single/two-step example
# Arguments:
#   parquet-file   Path to a Parquet file (use GenerateSampleParquetFileMain to create one)
#   column-name    Name of an integer column to filter on (e.g. zip_code)
#   int-literal    Integer threshold; rows where column-name > int-literal are kept
mvn exec:java \
  -Dexec.mainClass=ai.rapids.cudf.examples.HybridScanIoExample \
  -Dexec.args="/tmp/sample.parquet zip_code 150000"

# Chunked pipeline example
# Arguments:
#   parquet-file          Path to a Parquet file (use GenerateSampleParquetFileMain to create one)
#   row-group-batch-bytes Row group batch size in bytes: maximum total uncompressed size of the
#                         row groups in a single pass (batch). A pass is a batch of row groups
#                         whose combined uncompressed size fits within this limit.
#                         0 = no limit (all row groups in one pass).
#   chunk-bytes           Maximum size in bytes of a single output cuDF Table chunk within a pass.
#                         Controls how much decoded GPU memory is used at once per chunk.
#                         0 = no limit (entire pass materialised as one Table).
mvn exec:java \
  -Dexec.mainClass=ai.rapids.cudf.examples.HybridScanPipelineExample \
  -Dexec.args="/tmp/sample.parquet 67108864 16777216"
```

## Notes on filters

Because the Parquet schema may not be known to the application at AST construction time,
filter expressions for the hybrid scan reader use
`ai.rapids.cudf.ast.ColumnNameReference` to refer to columns by name (see the example
source). The reader resolves the names against the file schema during materialization.
