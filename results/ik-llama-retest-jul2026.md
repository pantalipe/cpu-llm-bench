# ik_llama.cpp retest — July 2026 checkout (build @ c2b58f8)

Same hardware (Xeon E3-1230 v2, pre-AVX2). Local repo already up to date with
`origin/main` @ `c2b58f8` (2026-07-09) — significantly newer than the original
test; `ggml/src/iqk/iqk_quantize.cpp` has grown to ~10,500 lines (was much
smaller at the time of the original test), i.e. this file has clearly seen
heavy rework since.

## cstdint fix — still needed, still applies cleanly

The originally-documented `iqk_common.h` missing-`<cstdint>` bug is still
present upstream. The existing local patch (`patches/ik_llama-cstdint-include.patch`)
still applies and is still required to get past the `popcount`/`uint8_t`
errors from the first build attempt.

## The old GGML_FP16_TO_FP32 bug is gone — but two NEW build errors replace it

With the cstdint patch applied, build now fails with different errors in
`ggml/src/iqk/iqk_quantize.cpp` (a file that has grown substantially since
the original test — this is likely why the specific bug shifted):

**Bug A — AVX2 intrinsics compiled outside an AVX2 guard (line ~1132):**
```
error: use of undeclared identifier 'hsum_i32_8'
```
This is inside a block using `__m256i` (AVX2-width) intrinsics
(`_mm256_cvtps_epi32`, `_mm256_add_epi32`) that gets compiled even on this
AVX1-only target — meaning either the surrounding `#if defined(__AVX2__)`
guard is missing/wrong for this specific function, or `hsum_i32_8` itself
is only defined inside an AVX2-gated section elsewhere in the file.

**Bug B — incomplete fallback path, literal `// TODO` stub (line ~7172):**
```
error: use of undeclared identifier 'y'
```
The non-NEON branch of an `#if ... #else` block is a `// TODO` comment
followed by code that references a `y[ib]` array never declared in that
branch's scope — i.e., the "generic CPU" fallback for this function was
never actually finished. This is exactly the code path a pre-AVX2, non-ARM
CPU should hit, and it's an unfinished stub, not a subtle bug.

## Takeaway

Same overall conclusion as the original test (build still fails on this
hardware), but the specific failure has moved — `ik_llama.cpp`'s IQK kernel
file is under active, fast-moving rework, and its coverage of the
"neither AVX2 nor ARM" code path is currently incomplete (bug A) and in at
least one place literally unimplemented (bug B, a `// TODO` stub). This
reinforces the original conclusion: this project's non-AVX2 x86 path is
under-tested by maintainers, to the point of having unfinished code in it.

Not fixed in this retest — bug B in particular isn't a config/include fix
like the cstdint patch, it needs someone to actually write the missing
scalar fallback logic.
