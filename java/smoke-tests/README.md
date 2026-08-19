# cudf-java smoke tests

Smoke test published `ai.rapids:cudf` classifier JARs on a Maven Central-style
classpath (cudf + slf4j only), the same way a consumer would depend on them.
Use this to check that packages load and run across the RAPIDS support matrix:
**OS × CUDA version × machine arch**.

## `run.sh` options

Exactly one source mode is required:

```bash
# Local gather tree (must contain ai/rapids/cudf/)
./java/smoke-tests/bin/run.sh --maven-repo-dir /tmp/cudf-java-build/maven-repo

# GitHub Actions gather artifact
./java/smoke-tests/bin/run.sh \
  --artifact-url 'https://github.com/rapidsai/cudf/actions/runs/<run>/artifacts/<id>'

# Installed into ~/.m2
./java/smoke-tests/bin/run.sh --use-maven-home

# Maven Central release (version required)
./java/smoke-tests/bin/run.sh --use-maven-central --version 25.12.0

# Narrow to one CUDA major
./java/smoke-tests/bin/run.sh --maven-repo-dir /tmp/maven-repo --cuda-version 12
```

| Flag | Meaning |
|---|---|
| `--artifact-url URL` | Fetch a java-gather artifact via [`bin/fetch_bundle.sh`](bin/fetch_bundle.sh) (authenticated `gh`). Cached under `.cache/downloads/` (or `$CUDF_JAVA_SMOKE_OUTPUT_DIR`); reused if already valid. |
| `--maven-repo-dir DIR` | Local Maven tree containing `ai/rapids/cudf/` |
| `--use-maven-home` | Use `~/.m2/repository` |
| `--use-maven-central` | Resolve from Maven Central (`--version` required) |
| `--version VER` | Pin `ai.rapids:cudf`. Required for Central; otherwise, if omitted, exactly one version directory must exist under `ai/rapids/cudf/`. |
| `--cuda-version 12\|13` | Narrow classifiers to this CUDA major |

Default classifiers by host arch:

| Arch | Default | With `--cuda-version 12` | With `--cuda-version 13` |
|---|---|---|---|
| `x86_64` | `unclassified`, `cuda12`, `cuda13` | `unclassified`, `cuda12` | `cuda13` |
| `aarch64` | `cuda12-arm64`, `cuda13-arm64` | `cuda12-arm64` | `cuda13-arm64` |

Missing JARs are warned and skipped. The run fails if no classifier receives a
smoke test, or if any classifier smoke test fails.

## Docker images

`run.sh` always runs inside Docker (CUDA runtime + OpenJDK 17 + Maven) covering
the [RAPIDS platform support](https://docs.rapids.ai/platform-support/) matrix:

- **OS:** Ubuntu 22.04, Ubuntu 24.04, Ubuntu 26.04, Rocky Linux 8
- **CUDA:** 12, 13
- **Arch:** `x86_64`, `aarch64`

Images under `docker/` are built on first use; rebuild with `docker build` /
`docker rmi` when a Dockerfile changes.
