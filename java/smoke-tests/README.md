# cudf-java smoke tests

Smoke test published `ai.rapids:cudf` classifier JARs on a Maven Central-style
classpath (cudf + slf4j only), the same way a consumer would depend on them.
Use this to check that packages load and run across the RAPIDS support matrix:
**OS × CUDA version × machine arch**.

The main test is [`SmokeTestCudf`](src/main/java/ai/rapids/cudf/smoke_test/SmokeTestCudf.java).

## `run.sh` options

Exactly one source mode is required:

```bash
# Local gather tree (must contain ai/rapids/cudf/)
./java/smoke-tests/bin/run.sh --maven-repo-dir /tmp/cudf-java-build/maven-repo

# GitHub Actions artifact (java-build, java-gather, or signed Maven Central bundle)
./java/smoke-tests/bin/run.sh \
  --artifact-url 'https://github.com/rapidsai/cudf/actions/runs/<run>/artifacts/<id>'

# Installed into ~/.m2
./java/smoke-tests/bin/run.sh --use-maven-home

# Maven Central release (version required)
./java/smoke-tests/bin/run.sh --use-maven-central --version 25.12.0

# Narrow to one CUDA major
./java/smoke-tests/bin/run.sh --maven-repo-dir /tmp/maven-repo --cuda-version 12

# Narrow to a subset of OSes
./java/smoke-tests/bin/run.sh --maven-repo-dir /tmp/maven-repo --os ubuntu24.04
```

| Flag | Meaning |
|---|---|
| `--artifact-url URL` | GitHub Actions artifact URL |
| `--maven-repo-dir DIR` | Local Maven tree containing `ai/rapids/cudf/` |
| `--use-maven-home` | Use `~/.m2/repository` |
| `--use-maven-central` | Resolve from Maven Central (`--version` required) |
| `--version VER` | Pin `ai.rapids:cudf`. Required for Central; otherwise, if omitted, exactly one version directory must exist under `ai/rapids/cudf/`. |
| `--cuda-version 12\|13` | Narrow classifiers to this CUDA major |
| `--os LIST` | Comma-separated OS list. Default: `ubuntu22.04,ubuntu24.04,ubuntu26.04,rockylinux8` |

Default classifiers by host arch:

| Arch | Default | With `--cuda-version 12` | With `--cuda-version 13` |
|---|---|---|---|
| `x86_64` | `unclassified`, `cuda12`, `cuda13` | `unclassified`, `cuda12` | `cuda13` |
| `aarch64` | `cuda12-arm64`, `cuda13-arm64` | `cuda12-arm64` | `cuda13-arm64` |

Missing classifier JARs and unsupported (classifier × OS) pairs are warned and
skipped. The run fails if no combination receives a smoke test, or if any smoke
test fails.

## Docker images

`run.sh` always runs inside Docker (CUDA runtime + OpenJDK 17 + Maven), building
one image per (CUDA major × OS) cell of the [RAPIDS platform support](https://docs.rapids.ai/platform-support/)
matrix:

| OS | CUDA 12 base image | CUDA 13 base image |
|---|---|---|
| `ubuntu22.04` | `nvidia/cuda:12.9.1-runtime-ubuntu22.04` | `nvidia/cuda:13.0.1-runtime-ubuntu22.04` |
| `ubuntu24.04` | `nvidia/cuda:12.9.1-runtime-ubuntu24.04` | `nvidia/cuda:13.0.1-runtime-ubuntu24.04` |
| `ubuntu26.04` | *(skipped — no official image)* | `nvidia/cuda:13.3.1-runtime-ubuntu26.04` |
| `rockylinux8` | `nvidia/cuda:12.9.1-runtime-rockylinux8` | `nvidia/cuda:13.0.1-runtime-rockylinux8` |
