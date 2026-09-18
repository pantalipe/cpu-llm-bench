# BitNet.cpp retest — March 2026 build (post 01/15/2026 CPU optimization update)

Same hardware as original test (Xeon E3-1230 v2, pre-AVX2, 16GB RAM).
Local `bitnet-test` repo rebuilt clean from `main` @ `01eb415` (2026-03-10),
which includes the "01/15/2026 BitNet CPU Inference Optimization" update.
`3rdparty/llama.cpp` submodule still pinned to `1f86f058` (unchanged) with
the `chrono` include patch already applied locally.

## Generation quality — still broken

`llama-cli.exe -m ggml-model-i2_s.gguf -p "Hi" -n 8 -t 4 --temp 0`

Same pre-tokenizer warning as before:
```
llm_load_vocab: missing pre-tokenizer type, using: 'default'
llm_load_vocab: GENERATION QUALITY WILL BE DEGRADED!
```

Output: `!!!!!!!!` (identical failure mode to the original test — NOT fixed
by the Jan 2026 update).

Eval speed: 6.44 tok/s (155.29 ms/token) — consistent with the original
6.28-6.46 t/s from `llama-bench`. Throughput gain holds; quality bug persists.

## New regression: llama-bench.exe crashes immediately

`llama-bench.exe -m ggml-model-i2_s.gguf -p 512 -n 128 -t 4`

Exit code `3221225725` (0xC00000FD, STACK_OVERFLOW) in under 2 seconds,
**before any output** — no stdout, no stderr, doesn't even reach model
loading. This is a regression vs. the original test, where `llama-bench`
completed successfully with these exact parameters and produced the
pp512/tg128 numbers in the original TL;DR table.

`llama-cli.exe --version` and `llama-cli.exe -m ... -p "Hi" -n 8` both run
fine on the same build — the crash is specific to `llama-bench.exe`, not a
general binary/environment issue.

**Not yet root-caused.** Candidate next steps: bisect which commit since
the original test broke `llama-bench` specifically, or run under a debugger
to get a stack trace (currently no output at all, even to stderr).

## Takeaway

The Jan 2026 "CPU Inference Optimization" update (parallel kernels, tiling,
embedding quantization — see upstream `src/README.md`) does not appear to
touch the pre-tokenizer/vocab handling path that causes the degenerate
output. Confirmed: same bug, same hardware, 2 months later. A new,
separate regression in `llama-bench.exe` was found in the process.
