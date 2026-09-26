# EXL3 in SGLang on one Intel Arc Pro B70 (campaign log)

Goal (Sero, 2026-09-26): serve EXL3 in SGLang on the free B70 and tune it to a protocol-accepted configuration
(`~/.claude/skills/inference-tuning-protocol`). Comparison bar: the vLLM exl3xpu recipe on the same card
(image 21412bdd: C1 thinking-on prose 91.2 / code 66.1 realistic, C16 ~365 agg, cold prefill 2,421 / 2,259 / 1,415
at 4K / 32K / 128K).

Card: B70 at PCI c3:00.0 = /dev/dri/renderD132 (card5) only. The other B70 (84:00.0, renderD134) serves
`exl3eval` and the Omarchy gateway and is not touched. Box work dir `omarchy:~/sglb70/`, containers/tmux `sglb70-*`.
Note: c3:00.0's slot (root port c0:01.1) has dropped off the bus 5 times in 3 days (PROGRESS.md); a drop
invalidates the cell in flight.

Approach: the exl3xpu ESIMD/XMX op library (`torch.ops.exl3xpu_C.linear`) behind SGLang's quantization interface,
registered through the `sglang.srt.plugins` entry point (`exl3xpu/sglang_plugin.py`, same shape as
0xSero/sglang-exl3 on CUDA). Base image `lmsysorg/sglang:v0.5.20-xpu`.

## Log

### 2026-09-26
- 01:22 (box clock) recon: renderD132 idle (gt idle residency climbing 1 s/s, no container maps it). Pull of
  `lmsysorg/sglang:v0.5.20-xpu` started (tmux `sglb70-pull`). Plugin `exl3xpu/sglang_plugin.py` written: per-shard
  loaders keyed on SGLang shard ids (q/k/v, 0/1, (0,1,2)/3), groups laid out exactly as the vLLM plugin
  (trellis concatenated along n, suh stacked per group, shard_of_nb), K/codebook read from safetensors headers,
  MTP `fc` served as EXL3, draft embed shared, optional pruned draft head (EXL3_DRAFT_VOCAB).
- Build in `lmsysorg/sglang:v0.5.20-xpu` (sglang 0.5.20, torch 2.13.0+xpu, oneAPI 2026.0 icpx, sglang-kernel-xpu 0.2.0):
  `scripts/build_ext.sh` 39 s; no `/opt/intel/oneapi/dnnl` in this image, so the oneDNN int8-prefill GEMM and the
  oneDNN fused SDPA are not built (fp16 prefill path). **Gate A1 PASS** on this stack (401 tensors, 25.6 G weights,
  every kernel path bit-identical to exllamav3 reconstruct).
- SGLang XPU constraints found at boot: `extra_buffer` mamba radix cache is refused on XPU; `no_buffer` needs
  page size 1 but the `intel_xpu` attention backend needs 64/128 -> with `intel_xpu` attention, prefix caching must
  be off (`--disable-radix-cache`). Decode graphs are opt-in (`--cuda-graph-backend-decode full`). SGLang 0.5.20 on
  XPU verifies speculative drafts greedily whatever the temperature (eagle_utils `... or _is_xpu`): fixed in the
  plugin (`_patch_xpu_spec_sampling`: sample the target token per verify row, greedy chain compare = lossless
  for a top-1 draft).
- **MILESTONE boot4 (00:55 container clock)**: EXL3 serves in stock SGLang 0.5.20 on the B70. 409 EXL3 linears
  (408 K=4 + K=6 head), 15.71 GB weights loaded in 6.5-14 s. Served id ok; thinking off -> "11:15"; thinking on
  -> reasoning separated, "11:15". Eager, no speculation, fp16 KV 173K tokens: **C1 prose 16.0 tok/s**, GPU 100 % busy.
- 01:57:26 box: **B70 c3:00.0 dropped off the bus again** (pciehp Slot(19) Link Down, MMIO 0xFFFFFFFF), during a
  C2 cell; the engine aborted in `drm_neo.cpp:284`. 6th drop in 3 days. Card re-enumerated as renderD133/card6;
  container `sglb70-dev` recreated on the new nodes. The C2 cell is invalid.
- 02:01:23 box: **7th link drop** (Slot 19), 90 s into the first graph-mode decode cell; corrected AER bursts on
  c0:01.1 line up with load (01:28-01:30 Gate A1, 02:00-02:01 decode) before each drop. Re-enumerated as
  renderD132. Launcher now recreates `sglb70-dev` automatically when the render node changes.
- boot5b: XPU decode graphs (`--cuda-graph-backend-decode full`, bs 1/2/4): **C1 prose 32.1 tok/s** (eager 16.0),
  i.e. at the no-draft linear-bound ceiling (~27 ms of EXL3 linears per token). (The first "graphs" attempt,
  boot5, never served: the tmux kill left boot4 running in the container; launcher fixed.)
- MTP bring-up (NEXTN 3/1/4): two more SGLang-XPU bugs fixed in the plugin: (1) the XPU copy of the fused
  sigmoid-gating delta-rule wrapper omits `stride_h0_source` (TypeError on the first verify) -> route to the generic
  wrapper; (2) `fused_mamba_state_scatter_with_mask` / `fused_conv_window_scatter_with_mask` refuse non-CUDA
  tensors (plain Triton kernels) -> guards rewritten to accept XPU. Draft `fc` served as EXL3, draft shares the
  target embed + head. Verify, draft-decode and draft-extend XPU graphs capture.
- **mtp3**: greedy output identical to the no-draft boot (same reasoning text, "11:15"). Synthetic panel, greedy,
  thinking off, no output cap, fp16 KV, 32K ctx, no prefix cache: **C1 prose 51.2, code 74.3 tok/s** per stream
  (accept len ~3.5-3.9 of 4 on code). vLLM exl3xpu, same synthetic panel: 56.6 / 79.2 (with its pruned draft head).
- Shared checkout collision: another agent committing in `~/intek-arc-b70` put its PROGRESS commit (8b49ebc,
  which also swept in this campaign's uncommitted sweep.py/pyproject.toml edits) on this branch. main was
  fast-forwarded to 8b49ebc (its only parent was main); this campaign continues in the worktree
  `~/intek-arc-b70-sglang` (branch `sglang-xpu`). Per that commit, B70 #0 (slot 17) also dropped at ~02:01.
- Pruned MTP draft head for SGLang (`_DraftHead` via `set_lm_head_from_target`, 512 of 1940 vocab blocks):
  C1 prose 51.2 -> **55.1** (+7.5 %), code 74.3 -> 76.9 (+3.5 %), greedy output unchanged. **Kept.**
- fp8 KV on the `intel_xpu` backend was half-wired in SGLang 0.5.20: (1) attention layers get no k/v scales
  unless the quant config provides a KV-cache method -> EXL3 config now returns `BaseKVCacheMethod` (scale 1.0, as
  vLLM's default fp8); (2) `forward_extend` (prefill chunks + speculative verify) hard-codes `k_descale=None` ->
  wired like decode; (3) `forward_decode` casts q to fp8, which the XPU kernel rejects -> q stays in the model
  dtype; (4) the fp8 prefill kernel needs a bf16 query -> the model runs `--dtype bfloat16` (EXL3 kernels take
  bf16 in/out; decode weights still bit-exact). Standalone kernel check (`tests/test_sgl_fa_fp8.py`): fp8 paged
  prefill with prefix correct (rel err <= 0.4 %), 4096 x 32K in 71 ms.
- GDN state: `--mamba-ssm-dtype float32` (config default) costs 2.4 GB state + 9.6 GB speculative intermediate
  states for 16 streams; float16 (what vLLM stores) halves it; output unchanged on the probe. XPU GDN verify
  wrapper now fixed in place (BV=16 kept) instead of routing to the generic BV=32 wrapper.
- "UR_RESULT_ERROR_OUT_OF_RESOURCES" during 4K-token prefill chunks at mem-fraction 0.88 (3.6 GB left) was
  device memory exhaustion, not the link: 0.80 passes the ladder 8K/16K/32K (1473 / 1419 / 1262 tok/s cold).
  Launcher bug found on the way: `pkill -f sglang.launch_server` inside `bash -c` matched itself (fixed).
- `--enable-linear-replayssm-spec` (drops the 4.8 GB per-draft state snapshots; KV pool would be 301K tokens at
  16 streams): **rejected** - its verify kernel fails in Intel Triton (tl.dot N>=16; padded to 16, then
  `TritonIntelGPURemoveLayoutConversions` pass failure with warps 1/4, tf32/ieee).
- Memory is the binding constraint: weights 15.9 GB + mamba (state + spec snapshots) 0.38 GB/stream + fp8 KV
  32 KB/token + ~4 GB activations. 16 streams cannot coexist with a 262K pool. **8 streams, mem 0.86: KV pool
  257,536 fp8 tokens**, ladder 8K/32K OK (s8a). This is the baseline config for the gate panel.
