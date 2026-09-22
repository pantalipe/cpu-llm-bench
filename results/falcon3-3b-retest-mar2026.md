# Falcon3-3B-Instruct-1.58bit retest — March 2026 build

Same build as `bitnet-2b-retest-mar2026.md` (bitnet-test @ 01eb415, 2026-03-10).

## Conversion — bug FIXED

Original test: HF→GGUF conversion crashed with a native access violation
(0xC0000005) at block 12/22, no traceback.

Retest: `utils/convert-hf-to-gguf-bitnet.py` completed successfully,
producing a 12.9GB f32 GGUF (`general.architecture = llama`, as expected —
Falcon3 uses a Llama-style arch name, unlike the base BitNet-b1.58 arch).
No crash, no errors. This bug is fixed since the original test.

## Quantization — works

`llama-quantize.exe ... I2_S` completed cleanly in ~165s, output 201/201
tensors, final size 2.06 GiB (5.49 BPW), model params 3.23B.

## Generation — new failure mode: silent/empty output

`llama-cli.exe -m ggml-model-i2_s.gguf -p "Hi" -n 8 -t 4 --temp 0`

No "GENERATION QUALITY WILL BE DEGRADED" pre-tokenizer warning this time
(unlike BitNet-2B). Model loads cleanly, sampler runs 8/8 eval steps
(190.53 ms/token, 5.25 tok/s), but **no generated text appears** in output
— just the echoed prompt ("Hi") followed directly by perf stats. Different
failure mode from BitNet-2B's `!!!!!!!!` degenerate output: here the model
appears to generate 8 tokens that produce no visible/printable text at all.

Not yet root-caused — candidates: BOS/EOS/PAD all mapped to token 11
(`<|endoftext|>`) and EOG token is also 11, which is unusual and could be
causing the sampler to emit only special/control tokens. Worth checking
with `--verbose-prompt` or a raw token dump next session.

## Takeaway

Progress since original test: the HF→GGUF conversion crash for Falcon3-3B
is fixed. But a working end-to-end pipeline is still not confirmed — the
model now converts and quantizes cleanly, only to hit a new, different
generation failure. Two out of three BitNet.cpp pipeline stages (convert,
quantize) now work for this model; generation still doesn't produce usable
output, just via a different mechanism than before.
