# Handoff: Java NVRTC JIT (`26.06` vs `26.08.1`) on x86

Continue this investigation on an **x86_64** host. 26.06 Java has **no ARM
classifier**; comparing jitify (26.06) vs rtcx (26.08) on the same classifier
requires x86 Maven Central JARs (`cuda13`, optionally `cuda12` / unclassified).

This is a **Java Maven Central consumer** issue. Do not bring in Spark RAPIDS.

Keep replies short. Do not reopen the “`.so.12` vs nothing” wording.

## Starter prompt for the next Cursor chat

```
Continue the Java cudf NVRTC JIT investigation from java/smoke-tests/JIT_NVRTC_INVESTIGATION.md.
This is an x86_64 host so we can test Maven Central 26.06 (no ARM classifier).
Java consumer only — not Spark RAPIDS. Keep answers to the point.

Goal: with NVRTC hidden (--hide-nvrtc), does 26.06 cuda13 fail the same way as
26.08.1 cuda13 on AST computeColumn? Run the matrix in that file and report.
```

## Symptom

Consumer error on **26.08.1** `cuda13` (aarch64 repro; same text expected on x86):

```
ai.rapids.cudf.CudfException: RTCX failure at: .../rtcx.cpp:285:
Failed to load dynamic library `libnvrtc.so` (tried: libnvrtc.so.13)
  at CompiledExpression.computeColumn(Native Method)
```

They report it appeared after a **version bump**. Question for x86: would
**26.06 `cuda13`** hit the same failure with NVRTC hidden?

## Facts (do not re-litigate)

- The JAR **CUDA classifier** picks the NVRTC soname: `cuda12` → `libnvrtc.so.12`,
  `cuda13` → `libnvrtc.so.13`.
- **Static libcudf** means libcudf + rtcx (26.08) or jitify (26.06) are in the
  JNI `.so`. It does **not** include NVRTC. Both backends `dlopen` NVRTC at JIT.
- `LIBCUDF_JIT_ENABLED` default is **`false`** on 26.06 and on `release/26.08`
  (26.08.1 = that branch after the hotfix merge; git tag `v26.08.01`; Maven
  version **`26.08.1`**).
- Java `CompiledExpression.computeColumn` calls `cudf::compute_column`, which
  takes the JIT path only if `get_context().use_jit()`.
- 26.08 Java `computeColumnJit` exists on **main after** 26.08.1; the Central
  26.08.1 JAR only has `computeColumn`.
- **Do not** claim 26.06 skipped NVRTC. Jitify loads it dynamically (see
  pointers below).

## Tooling on this branch

| Piece | Role |
|---|---|
| `java/smoke-tests/bin/run.sh` | Docker matrix runner. Forwards `LIBCUDF_JIT_ENABLED` if set on the host. `--hide-nvrtc` builds `*:*-nonvrtc`. |
| `java/smoke-tests/bin/cuda_os_docker.sh` | `ensure_image_without_nvrtc`: **move** `libnvrtc*` to `/opt/nvrtc-hidden`, then `ldconfig`. |
| `java/smoke-tests/src/main/java/.../SmokeTestCudf.java` | Step **AST computeColumn**: `new ColumnReference(0).compile().computeColumn(table)`. |

**Do not rename `libnvrtc*` in place.** `ldconfig` recreates `libnvrtc.so.13`
and `dlopen` still succeeds. Verify hide:

```bash
docker run --rm cudf-java-smoke-test:cuda13-ubuntu22.04-nonvrtc \
  bash -c 'ldconfig -p | grep nvrtc || echo NONE'
```

Expect `NONE`. If the image was built with the old rename hack, `docker rmi`
the `*-nonvrtc` tag and rebuild.

CUDA **runtime** images **do** ship NVRTC (seen on
`nvidia/cuda:13.0.1-runtime-ubuntu22.04` aarch64:
`/usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvrtc.so.13`). Hiding is
required to reproduce a consumer host without NVRTC.

## x86 run matrix

Host: x86_64, Docker with `--gpus all`, this git branch.

Confirm Maven versions on Central (`ai.rapids:cudf`). 26.08.1 is `26.08.1`.
26.06 is likely `26.06.0` — check before running.

Narrow to one OS while iterating: `--os ubuntu22.04`.

```bash
cd <cudf-checkout>

# 1) Control: 26.08.1 cuda13, NVRTC present, JIT on → expect PASS
LIBCUDF_JIT_ENABLED=true ./java/smoke-tests/bin/run.sh \
  --use-maven-central --version 26.08.1 --cuda-version 13 --os ubuntu22.04

# 2) Repro: 26.08.1 cuda13, NVRTC hidden, JIT on → expect RTCX / libnvrtc.so.13
LIBCUDF_JIT_ENABLED=true ./java/smoke-tests/bin/run.sh \
  --use-maven-central --version 26.08.1 --cuda-version 13 --os ubuntu22.04 \
  --hide-nvrtc

# 3) Interpreter: 26.08.1 cuda13, NVRTC hidden, JIT explicitly off
#    Expected: PASS if computeColumn stays off the rtcx path.
#    aarch64: JIT unset + hide-nvrtc still failed at AST (see Open). Repeat
#    here with false, then with the var unset.
LIBCUDF_JIT_ENABLED=false ./java/smoke-tests/bin/run.sh \
  --use-maven-central --version 26.08.1 --cuda-version 13 --os ubuntu22.04 \
  --hide-nvrtc

env -u LIBCUDF_JIT_ENABLED ./java/smoke-tests/bin/run.sh \
  --use-maven-central --version 26.08.1 --cuda-version 13 --os ubuntu22.04 \
  --hide-nvrtc

# 4) The 26.06 question: same hide + JIT on, cuda13
LIBCUDF_JIT_ENABLED=true ./java/smoke-tests/bin/run.sh \
  --use-maven-central --version 26.06.0 --cuda-version 13 --os ubuntu22.04 \
  --hide-nvrtc
```

If (4) fails, expect a **jitify** error (not `RTCX failure`), still looking
for `libnvrtc.so.13`. That answers: yes, 26.06 cuda13 also needs NVRTC when
JIT runs.

Optional: 26.06 `--cuda-version 12` with `--hide-nvrtc` (looks for `.so.12`).
x86 default classifiers include `unclassified` (cuda12-class) plus `cuda12`.

Logs: `java/smoke-tests/.cache/logs/smoke-test-<classifier>-<os>[-nonvrtc].log`

## aarch64 results already obtained (26.08.1 `cuda13-arm64`, ubuntu22.04)

| Command | Result |
|---|---|
| JIT unset, NVRTC present | PASS (interpreter; runtime image had `libnvrtc.so.13`) |
| `LIBCUDF_JIT_ENABLED=true`, NVRTC present | PASS |
| `LIBCUDF_JIT_ENABLED=true`, `--hide-nvrtc` | **FAIL** — exact consumer RTCX `libnvrtc.so.13` message |
| JIT unset, `--hide-nvrtc` | **FAIL** at AST with the same RTCX message (unexpected if default is false) |
| `LIBCUDF_JIT_ENABLED=false`, `--hide-nvrtc` | **not finished** (run interrupted) |

## Open questions for x86

1. Same `--hide-nvrtc` + JIT on: does **26.06 `cuda13`** fail (jitify) like
   **26.08.1 `cuda13`** (rtcx)?
2. Why did aarch64 fail with JIT **unset** + hide-nvrtc? Repeat with
   `LIBCUDF_JIT_ENABLED=false` (must be set; `run.sh` only injects the env if
   it is present on the host).
3. If interpreter still loads NVRTC on 26.08.1, that is a 26.08 behavior
   change vs “JIT off needs no NVRTC”, not just a soname mismatch.

## Code pointers

**26.08 / 26.08.1 (`release/26.08`):**

- `cpp/src/runtime/context.cpp` — `LIBCUDF_JIT_ENABLED` default false;
  `rtcx::initialize()` in `ensure_jit_cache_initialized()`.
- `cpp/src/transform/compute_column.cu` — `use_jit()` → `compute_column_jit`.
- `cpp/src/jit/cache.cpp` — `rtcx::compile`.
- `cpp/cmake/thirdparty/get_rtcx.cmake` — librtcx
  `GIT_TAG efad266c1fd9de6d8486c6ba71bfa74df063eb1f`.
- NVRTC `dlopen` is **inside librtcx** (`rtcx.cpp` ~285), not in cudf sources.

**26.06 jitify NVRTC `dlopen`:**

- Pin: `v26.06.00` `cpp/cmake/thirdparty/get_jitify.cmake` →
  `https://github.com/NVIDIA/jitify.git` @ `44e978b21fc8bdb6b2d7d8d179523c8350db72e5`
- [`jitify2.hpp`](https://github.com/NVIDIA/jitify/blob/44e978b21fc8bdb6b2d7d8d179523c8350db72e5/jitify2.hpp):
  `JITIFY_LINK_NVRTC_STATIC` defaults to `0`; `DynamicLibrary::open` calls
  `dlopen`; `LibNvrtc` ctor tries `libnvrtc.so.<CUDA_VERSION major>` then
  `libnvrtc.so.<major>.<minor>` for minor 9..0.

**26.08 PRs (not listed as breaking):** #22654 (jitify → rtcx), #22823
(static rtcx deps; NVRTC still dynamic), #23089 (Python wheel ships NVRTC;
Java JAR does not).

## Constraints

- Do not commit unless asked.
- `--hide-nvrtc` is a **repro hack**, not CI default.
- Rebuild `*-nonvrtc` images after changing `ensure_image_without_nvrtc`.
