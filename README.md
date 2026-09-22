# CPU LLM Bench — testing sub-2-bit and CPU-optimized LLM inference on old, AVX2-less hardware

Most "run an LLM on CPU" benchmarks out there are done on modern hardware (AVX2/AVX-512, Apple Silicon).
This repo documents what actually happens when you try the current generation of CPU-first inference
projects (Microsoft's BitNet, ikawrakow's ik_llama.cpp) on a **pre-AVX2 CPU** — the kind of box a lot of
homelabbers and small dev shops actually have sitting around.

No sponsorship, no cherry-picking — this is a real test log, bugs and all, kept as a public reference
for anyone in the same boat.

## Hardware

- CPU: Intel Xeon E3-1230 v2 (Ivy Bridge, 4C/8T, 3.30GHz) — **no AVX2, no FMA, no AVX-512** (AVX1 only)
- RAM: 16 GB
- GPU: NVIDIA GT 730 (OpenGL 3 only — no CUDA, no Vulkan) — not usable for inference
- OS: Windows 10
- Baseline LLM stack: llama-swap + Ollama, `qwen3:14b` (Q3_K_XL), CPU-only

This CPU predates AVX2 (introduced with Haswell, 2013). A huge share of the "150-350% CPU speedup" claims
from newer inference projects assume AVX2 or better is available — worth keeping in mind if you're on
similar-era hardware.

## TL;DR

| Test | Result |
|---|---|
| Current baseline (`qwen3:14b`, Ollama, CPU) | 1.2 tok/s |
| BitNet-b1.58-2B-4T (official microsoft/BitNet, I2_S, 4 threads) | **6.3-6.5 tok/s** (~5x) — speed confirmed |
| BitNet-b1.58-2B-4T — actual text generation | ❌ broken (see bugs below) |
| Falcon3-3B-Instruct-1.58bit (official BitNet, HF→GGUF conversion) | ❌ crash mid-conversion (fixed as of Mar 2026 retest — see below) |
| ik_llama.cpp build (AVX1-only x86 target) | ❌ build fails, two distinct bugs found |
| Qwen3-14B IQ3_XXS vs Q3_K_XL (same model, llama.cpp mainline) | **no speed gain** (1.12-1.22 vs 1.2 tok/s) — I-quants need AVX2 to pay off |
| Qwen3-14B IQ3_XXS via llamafile 0.10.0 (same GGUF, same hardware) | **~5x slower** than mainline llama.cpp (~0.21 vs 1.12 tok/s) |

**Conclusion so far:** the raw speed claim for CPU-only ternary/low-bit inference checks out — on hardware
with zero SIMD advantages beyond AVX1, BitNet.cpp still delivered ~5x the throughput of a conventional
Q3_K_XL 14B model on llama.cpp/Ollama. But neither of the two projects tested produces a *working, complete*
pipeline out of the box on this hardware today — both hit real, reproducible bugs before a clean end-to-end
generation was possible. Details and patches below.

## Update — March 2026 retest (post Jan-2026 BitNet CPU optimization)

Microsoft shipped a "BitNet CPU Inference Optimization" update on 01/15/2026. Retested on the same
hardware with `bitnet-test` rebuilt from `main` @ `01eb415` (2026-03-10). Full details in
`results/bitnet-2b-retest-mar2026.md` and `results/falcon3-3b-retest-mar2026.md`.

| Test | Result |
|---|---|
| BitNet-2B generation quality bug | still broken — same pre-tokenizer warning, same `!!!!!!!!` output |
| BitNet-2B eval speed | 6.44 tok/s — throughput gain still holds |
| `llama-bench.exe` | new regression — crashes instantly (stack overflow, 0xC00000FD), zero output |
| Falcon3-3B HF→GGUF conversion | fixed — completes cleanly, no more block-12 crash |
| Falcon3-3B I2_S quantization | works — 201/201 tensors, 2.06 GiB final size |
| Falcon3-3B generation | new failure mode — no visible output text at all (possibly BOS/EOS/PAD/EOG all mapping to token 11) |

**Bottom line:** 2 months of upstream fixes moved the needle on Falcon3-3B (conversion + quantization
now work) but generation still isn't usable for either model, via two different failure modes. Net new
finding: a `llama-bench` regression that wasn't present in the original test.

## Methodology

- Speed: `llama-bench` (or the project's own benchmark harness), pp512/tg128, default settings unless noted.
- Baseline comparison: [ollama-bench](https://github.com/pantalipe/ollama-bench), a small throughput/consistency
  harness against an OpenAI-compatible endpoint (llama-swap), same machine, same task set.
- All models tested at their smallest/officially-supported size first, to isolate whether an issue is about
  raw compute limits vs. actual bugs.

## 1. microsoft/BitNet (bitnet.cpp)

Repo: https://github.com/microsoft/BitNet — pinned to an older llama.cpp fork commit (`1f86f058`).

### Build

Required Visual Studio 2022 Build Tools with the Clang/LLVM toolset (`Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset`)
— the LLVM binaries alone are **not** sufficient; you need the MSBuild integration component too, or `cmake -T ClangCL`
fails at the generate step.

**Bug found:** the vendored llama.cpp fork is missing `#include <chrono>` in 4 files, which fails on
Clang 19 (works by accident on some GCC/older Clang setups that transitively pull it in elsewhere):
- `common/common.cpp`
- `common/log.cpp`
- `examples/imatrix/imatrix.cpp`
- `examples/perplexity/perplexity.cpp`

Fix: add `#include <chrono>` near the other standard includes in each file. See `patches/bitnet-chrono-includes.patch`.

After that, build succeeds cleanly (`llama-cli`, `llama-bench`, `llama-quantize`, `llama-server`).

### Speed benchmark

Official `BitNet-b1.58-2B-4T` GGUF (from `microsoft/BitNet-b1.58-2B-4T-gguf`, pre-quantized I2_S, ~1.1GB):

```
llama-bench -m ggml-model-i2_s.gguf -p 512 -n 128 -t 4
| model                                 | size    | params | test  | t/s          |
| bitnet-b1.58 2B I2_S - 2 bpw ternary  | 1.71 GiB | 2.74 B | pp512 | 6.28 ± 0.19  |
| bitnet-b1.58 2B I2_S - 2 bpw ternary  | 1.71 GiB | 2.74 B | tg128 | 6.46 ± 0.27  |
```

At `-t 8` (all logical threads, incl. hyperthreads): pp512 dropped slightly (5.55 t/s), tg128 about the
same (6.73 t/s) — on a 4-core/8-thread CPU, using only the 4 physical cores was as good or better than
using all 8 logical threads for this workload.

### Actual generation — broken

Loading the official pre-converted GGUF prints:

```
llm_load_vocab: missing pre-tokenizer type, using: 'default'
llm_load_vocab: ************************************
llm_load_vocab: GENERATION QUALITY WILL BE DEGRADED!
llm_load_vocab: CONSIDER REGENERATING THE MODEL
llm_load_vocab: ************************************
```

With greedy decoding (`--temp 0`) on a trivial prompt ("Hello, how are you?"), output degenerated into
repeated `!!!!!!!!!!` tokens — model loads fine, inference runs, but the pre-tokenizer mismatch between
this llama.cpp fork's vocab handling and the officially-published GGUF makes output unusable.

On a longer, structured prompt (~15 lines, a commit-message-generation task), `llama-cli` crashed outright
with a **stack overflow** (exit code `0xC00000FD`) partway through generation. Reproduced twice.

### Falcon3-3B via BitNet — different, harder failure

Tried converting `tiiuae/Falcon3-3B-Instruct-1.58bit` from HF safetensors instead (rather than using a
pre-made GGUF), since the official BitNet repo ships a conversion script for this. Two failures encountered
in sequence:

1. The *current* HF repo for the official 2B-4T model uses `BitNetForCausalLM` as its `architectures` field;
   the bundled `utils/convert-hf-to-gguf-bitnet.py` in this llama.cpp fork doesn't recognize that architecture
   name (`NotImplementedError: Architecture 'BitNetForCausalLM' not supported!`). Not an issue for Falcon3
   itself (uses a Llama-style architecture name) but worth knowing if you try the base model this way.
2. For Falcon3-3B specifically: conversion crashed with a **hard native access violation** (`0xC0000005`,
   no Python traceback) partway through layer conversion (crashed at block 12 of 22). No exception, no
   graceful failure — looks like a torch/numpy interaction issue specific to this environment's package
   versions, unpacking the packed `uint8` ternary weight tensors.

## 2. ikawrakow/ik_llama.cpp

Repo: https://github.com/ikawrakow/ik_llama.cpp — actively maintained llama.cpp fork, advertised
"first-class Bitnet support" and 150-350% CPU speedups via its IQK kernels. On Windows this builds with
plain MSVC (no Clang needed) according to upstream docs.

### Build attempt 1 — MSVC

Configured cleanly (correctly detected AVX1-only, no AVX2/FMA/AVX512 — no silent fallback issues). Build
failed hard in `ggml/src/iqk/*.cpp` — dozens of `error C2065: 'uint8_t': identifier not found` and
`error C2374: 'popcount': redefinition`.

**Root cause:** `ggml/src/iqk/iqk_common.h` has an `#if defined(_MSC_VER)` block (providing `popcount()`
via `__popcnt`/`_mm_popcnt_u64`) that assumes `<cstdint>` is already visible — it isn't, in a plain MSVC
build of this translation unit.

### Build attempt 2 — Clang (clang-cl)

Same error, because `clang-cl` also defines `_MSC_VER` (it's MSVC-command-line-compatible) — so this isn't
an MSVC-only bug, it hits any MSVC-ABI-compatible compiler.

**Fix:** add `#include <cstdint>` to that `_MSC_VER` block in `iqk_common.h`. See
`patches/ik_llama-cstdint-include.patch`.

### Build attempt 2, continued — second bug

After the above fix, hit a second, unrelated failure in `ggml/src/iqk/iqk_cpu_ops.cpp`: `GGML_FP16_TO_FP32`
and `ggml_half` reported as unknown identifiers, despite `ggml-impl.h` (which defines a portable fallback
for `GGML_FP16_TO_FP32` at file scope, unconditionally, via a lookup-table function) being included.

Did not get to the bottom of this one — it needs macro-expansion tracing (`clang -E`) to pin down exactly
which include path fails to bring in the definition for this specific translation unit on a non-ARM,
non-AVX2 target. Flagging it here in case it saves someone else the rediscovery. Not fixed in this repo.

## Repo layout

```
patches/    — actual diffs applied during testing (chrono includes, cstdint include)
results/    — raw benchmark JSON / logs
```

## Takeaways

- The core BitNet.cpp speed claim (CPU-only, no SIMD-advantage dependency) held up on genuinely old hardware:
  ~5x over a conventional Q3_K_XL 14B model on the same box.
- Neither project tested is currently a turnkey drop-in replacement on a pre-AVX2 x86 CPU — both have real,
  reproducible bugs specific to less-common build targets (old CPUs, non-ARM+non-AVX2 paths) that are
  probably under-tested by maintainers who mostly develop/test on AVX2+ or ARM hardware.
- If you're on similar hardware and want CPU speedup *today*, the pragmatic move is probably staying on
  mainline llama.cpp/Ollama with a smaller conventional model (e.g. a 3-4B Q4_K_M) rather than chasing
  bleeding-edge low-bit CPU kernels — until these forks get more non-AVX2 testing coverage.

## 3. Mainline llama.cpp + IQ quants (a common "just do this instead" suggestion)

A frequent recommendation once BitNet/exotic runtimes hit trouble: skip them and just use llama.cpp's
own low-bit "I-quants" (`IQ2_XS`, `IQ3_XXS`, etc.) on a conventional model. Worth testing directly, since
it needs zero patches — mainline `ggml-org/llama.cpp` built clean on this CPU with no changes required
(a good sign in itself, compared to the two forks above).

Tested `Qwen3-14B` — the exact same model already used as the baseline, just re-quantized — at `IQ3_XXS`
(bartowski/Qwen_Qwen3-14B-GGUF), against the existing `Q3_K_XL` baseline:

| Config | pp512 | tg128 |
|---|---|---|
| Qwen3-14B, Q3_K_XL (baseline, Ollama) | — | 1.2 tok/s |
| Qwen3-14B, IQ3_XXS (llama.cpp mainline, same hardware) | 1.22 tok/s | 1.12 tok/s |

**No speed gain at all** — essentially identical, if anything marginally worse for generation. This isn't
a bug, it's expected: I-quant kernels rely on SIMD tricks that need AVX2 to pay off. Without AVX2, you pay
the extra dequantization/compute cost of the fancier format without getting the offsetting speedup. This
matches bartowski's own disclaimer on quant model cards ("I-quants can also be used on CPU, but will be
slower than their K-quant equivalent").

**Takeaway:** on a pre-AVX2 CPU, switching quant *format* at the same parameter count buys you smaller
files, not speed. If BitNet doesn't work cleanly (see above) and IQ-quants don't help either, the only
remaining lever for raw throughput is reducing parameter count (a smaller model), which is orthogonal to
quantization scheme entirely.

## Contributing

If you've hit (or fixed) any of the above on similar hardware, open an issue or PR — especially interested
in a fix for the `ik_llama.cpp` `GGML_FP16_TO_FP32` issue.
