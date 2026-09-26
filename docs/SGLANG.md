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
- Gate panel on s8a (tmux `sglb70-panel`, real corpus, t=0.7/top-p 0.95, no output cap): C1 prose thinking on
  **55.6 tok/s per stream** (58.8 agg; 1 sample, the request outlived the 45 s window; MTP accept len median 2.50 over
  the boot), GPU 100 % busy. 02:45:00: **8th link drop** (Slot 19) 90 s into the C2 cell; the engine aborted
  (drm_neo.cpp:284); every later cell is REQ_FAIL (not a measurement). Card re-enumerated as renderD133.

### Stop: B70 #1 cannot carry sustained load, and its faults reach other devices
Kernel log correlation (journalctl -k), per drop window: 01:57 -> RxErr bursts on the RTX 3090 at c5:00.0
(34 lines, same root complex pci0000:c0); **02:01:23 -> B70 #0 (slot 17, 80:03.1) lost its link in the same second
as B70 #1** (the other agent's TB 2.1 eval engine on B70 #0 segfaulted and the user's live engine was recreated,
per its PROGRESS entry); 02:45 -> 92 RxErr lines on the 3090 at c5:00.0 plus c0:03.1. Idle: 0 errors in 90 s.
Under load the corrected-error rate on c0:01.1 climbs, then the link drops within 1.5-5 min (earlier this week it
took 20-25 min). Continuing would put the user's live service on B70 #0 and the 3090 campaign at risk, so GPU work
on c3:00.0 is halted. Needs a hardware/BIOS decision by the owner (reseat or replace the slot-19 riser, force
PCIe Gen3/Gen4 on the port, `pcie_aspm=off`, check the PSU rail both B70s share). No root here, so none of that
was attempted.

### Panel as measured (one B70, Qwen3.8-27B EXL3 4.00bpw, MTP k=3 + pruned draft head)
| cell | SGLang 0.5.20 + exl3xpu (this campaign) | vLLM 0.26.1 exl3xpu (image 21412bdd) |
|---|---|---|
| C1 prose, thinking on (real corpus, t=0.7) | **55.6** per stream (1 sample, s8a: bf16, fp8 KV, 262K ctx, 8 seqs) | 91.2 agg (accept 3.42) |
| C1 code, thinking on (real, t=0.7) | not measured (link drop) | 66.1 |
| C1 prose / code, thinking off (synthetic greedy) | **55.1 / 76.9** (mtp4: fp16, fp16 KV, 32K ctx, 4 seqs) | 56.6 / 79.2 (09-23 build) |
| C2 / C8 / C16 aggregate | not measured (link drop during C2) | 151.4 / 357.1 / 365.2 (prose, thinking on) |
| cold prefill 4K / 32K / 128K | ~1.2K (4K ladder rung) / **1,262-1,265** / not measured; 8K 1,394-1,473, 16K 1,419 | 2,421 / 2,259 / 1,415 (int8 prefill + oneDNN attention); fp16-prefill recipe 1,654 / 1,434 / 928 |
| max context / KV pool | 262,144 / 257,536 fp8 tokens at 8 seqs | 262,144 / 272,570 at 16 seqs |
Reading: at C1 SGLang is at parity with vLLM on the synthetic greedy thinking-off panel (97 %) but ~40 % behind on
the realistic thinking-on cell: per verify step 45 ms vs vLLM's 37.5 ms, and acceptance 2.5 vs 3.4 (not yet
explained: sampling path vs vLLM's rejection sampler, or the corpus sample; one sample only). Prefill without int8
is ~12 % below vLLM's fp16-prefill recipe at 32K and ~44 % below the current int8 + oneDNN recipe. SGLang holds half
the streams at the full context: the GDN state + per-draft snapshots cost 0.38 GB/stream and ReplaySSM (the fix)
does not compile on Intel Triton.

### Gate status (inference-tuning-protocol)
| gate | status |
|---|---|
| 1 correctness | served id, completions, thinking on/off, greedy identity MTP vs no-MTP: pass. Tool call, vision: **not run**. Coherence ladder: 1K/4K/8K/16K/32K OK; 128K/256K **not run** |
| 2 speed floor | C1 prose 55.6 >= 20: pass; >= 90 % of the vLLM recipe (91.2): **fail** (61 %) |
| 3 sustained (3x30 s cells, 10 min soak) | **not possible on this card** (link drops within minutes) |
| 4 context proof 32K | pass (32,688-token needle OK, TTFT 25.9 s); 128K not run |
| 5 headroom | 4.1 GB free after graphs at mem 0.86 (static 27.4 GB of 31.9) |
| 6 speculative tried | MTP k=3 on, kept (+7.5 % from the pruned head); draft-length sweep not run |
| 7 concurrency computed | 257,536 / 262,144 = 0.98 full-context streams; max_running_requests 8 |
Status: **candidate, tuned: baseline-partial**. No registry recipe was written: there is no published SGLang-XPU
image to pin and no panel to attach; a recipe from two cells would be the one-wave "validated" the protocol forbids.

### Next steps (on a stable B70)
1. Re-run `scripts/sglb70_panel.sh` on the s8a config (C1-C16 thinking on/off, prefill 4K/32K/128K), tool + vision
   checks, ladder to 256K, 10-min soak at C8.
2. int8 prefill: `_C_dnnl.so` (pip onednn-devel 2026.0.0, EXL3_DNNL_DIR) is built but untested; serve with
   `EXL3_LIB=/w/exl3xpu/exl3xpu/_C_dnnl.so EXL3_INT8_PREFILL=1` (vLLM measured +38 % at 32K from it).
3. Explain the acceptance gap (2.5 vs 3.4) with an A/B of `EXL3_SGL_SPEC_SAMPLE` at t=0 vs t=0.7 on the same prompts.
4. Knob ladder: draft length 2/3/4, chunked prefill 2048/4096/8192, mem fraction 0.86 -> 0.90 with chunk 2048,
   max-running 8 vs 12 at 196K context; triton attention backend + radix cache (prefix caching is off with intel_xpu).
5. Upstream: the five SGLang-XPU bugs fixed in the plugin (spec sampling greedy-only, stale XPU GDN wrapper,
   CUDA-only mamba scatter guards, fp8 descale in forward_extend, fp8 q cast in forward_decode) are worth issues/PRs.

### Box state at stop (2026-09-26 ~03:00 box clock)
Container `sglb70-dev` (lmsysorg/sglang:v0.5.20-xpu, `sleep infinity`, maps only renderD133/card6 = c3:00.0,
pristine venv + exl3xpu editable + onednn 2026.0.0 --no-deps); no SGLang server, no benchmark, no `sglb70-*` tmux
session. Work dir `~/sglb70/` (exl3xpu copy, runs/, logs/, byp/). Image `lmsysorg/sglang:v0.5.20-xpu` pulled.
Nothing else on the box was touched.

## 2026-09-26: second campaign on the stable B70 (84:00.0), goal SGLang >= vLLM on every cell
Sero approved borrowing B70 #0 (84:00.0, renderD135/card8). The live Omarchy engine
(`...-tp1-engine`, image ghcr.io/0xsero/exl3xpu@sha256:86276b00, fp16 prefill) was `docker stop`ped at 09:31 box
time after recording `docker inspect` to `omarchy:~/sglb70/live-engine-inspect.json` and a gateway chat check
("17*3" -> "51"). c3:00.0 is not used under load any more; all scripts default to 84:00.0.

### Paired vLLM baseline (vllm0926): image 21412bdd, same argv as the live engine, 84:00.0, today
Recipe model.yaml (MTP k=3 pruned head, fp8 KV, 262,144 ctx, 16 seqs, chunk 4096, int8 prefill, oneDNN attention),
prefix caching on. vLLM KV pool 310,472 tokens -> 1.18 full-context streams. Panel = `scripts/sglb70_panel.sh`:
real corpus, t=0.7 top-p 0.95, no output cap, 15 s warm + 45 s window, one wave; prefill 2 waves (first discarded).
| cell | think on: prose / code (agg, accept) | think off: prose / code (agg, accept) |
|---|---|---|
| C1 | 75.7 (3.05) / 63.4 (2.70) | 54.9 (2.29) / 83.0 (3.48) |
| C2 | 129.9 / 122.4 | 98.9 (LOOP flag) / 147.2 |
| C4 | 224.6 / 205.4 | - |
| C8 | 300.2 / 303.2 | 240.0 / 370.6 |
| C16 | 318.1 / 394.9 | 310.6 / 474.6 |
Cold prefill: 4K 2,461, 32K 2,227, 128K 1,359 tok/s.

### SGLang fixes on 84:00.0 (all paired against vllm0926, same card and day)
1. **Acceptance "gap" was a measurement artifact.** SGLang's `generation_tokens_total / spec_verify_calls_total`
   counters move only when a request finishes; thinking-on requests run 10-21K tokens, so a 45 s cell sees a biased
   few. Whole-request probe (`bench/accept_probe.py`, same 4 real prompts, t=0.7, thinking on, no cap):
   **SGLang 3.42 tokens/step** (59,468 tokens) = vLLM's 3.42 on its realistic panel. Sampling defaults are identical
   (generation_config: t 1.0, top-k 20, top-p 0.95; requests override t=0.7/top-p 0.95). Not a lever.
2. **Verify attention scaled with context (the real per-step gap at long outputs).** SGLang's step time grew
   3.5 ms per 1K tokens of context (47 ms at 0, 104 ms at 16K): the 4-query verify took sgl_kernel's prefill path,
   3.0 ms/layer at 16K fp8 keys vs 0.08 ms for a 1-query decode (`tests/bench_sgl_verify_attn.py`; 13.7 ms at 64K).
   Fix `EXL3_SGL_VERIFY_AS_DECODE` (default on): each verify row is a decode query over the same pages with
   cache_seqlens L-(nq-1-j) (exact bottom-right causal; no host sync, graph-captured). `tests/test_sgl_verify_as_decode.py`
   PASS (worst rel err 7e-3 vs the stock kernel = bf16 rounding, fp8 and bf16, mixed lengths to 40K). Step time now
   flat: 44.7 ms at 0, 46.0 at 4K (was 59.4). C1 prose thinking-on 82.6 agg (vLLM 75.7). **Kept.**
3. **Memory: "UR_RESULT_ERROR_OUT_OF_RESOURCES" crashes at C8/C16 and on the 2nd prefill chunk.** Causes found:
   (a) Inductor compile workers: up to 34 subprocesses each opened an XPU context (~0.18 GiB each, ~6 GiB transient)
   -> `TORCHINDUCTOR_COMPILE_THREADS=1`; (b) SGLang's default `max_prefill_tokens` 16384 batches several prompts
   -> `--max-prefill-tokens 4096` (vLLM's 4096 budget); (c) only ~2 GiB of the card is really free after startup
   (scheduler holds 29.7 GiB of BOs, ~1.6 GiB outside torch), while a 4096-token chunk with a prefix takes ~2.2 GiB
   of fragmented torch reserve -> `PYTORCH_ALLOC_CONF=expandable_segments:True`; (d) the per-draft GDN state
   snapshots (see 4). Tools: `EXL3_MEM_TRACE`, `tests/vram_sampler.py`, `tests/mem_*.py`. Also kept: graph warm replay
   after capture (no measurable effect), `EXL3_TORCH_MEM_FRACTION` cap (not used).
4. **GDN ReplaySSM fold-every-commit for GDN** (`EXL3_SGL_GDN_FOLD=1` + `--enable-linear-replayssm-spec`): SGLang
   0.5.20 implements the raw-input fold but gates it to KDA; the compact circular GDN kernel does not compile on
   Intel Triton. Lifting the gate: per-draft snapshots 2.4 GB (8 streams) -> 0, rings 27 MB, verify via the generic
   fused recurrent kernel, commit via `gdn_replayssm_exact_fold_kernel` (no tl.dot). **KV pool 257,536 -> 341,376 fp8
   tokens (1.30 full-context streams; vLLM 1.18)**. Greedy texts differ from the snapshot path (fp16 state
   materialization + BV 32 vs 16), acceptance identical (2.33 greedy thinking-off). Coherence ladder
   1K/4K/8K/32K/**128K** OK. **Kept.**
Current config (sg14): base args (`scripts/sglb70_base_args.txt`: 262,144 ctx, mem 0.86, intel_xpu, fp16 mamba state,
no radix cache, 8 running, graphs 1-8, NEXTN 3/1/4, fp8 KV, bf16, metrics, max-prefill 4096)
+ `--enable-linear-replayssm-spec`; env EXL3_DRAFT_VOCAB, TORCHINDUCTOR_COMPILE_THREADS=1, EXL3_SGL_GDN_FOLD=1,
PYTORCH_ALLOC_CONF=expandable_segments:True. Prefill (ladder, cold): 8K 1,476, 32K 1,262, 128K 756 tok/s.

### Prefill levers (paired, cold ladder prompts, 84:00.0)
| step | 4K | 32K | 128K | kept |
|---|---|---|---|---|
| sg14 (fold + expandable segments), fp16 prefill, sgl fp8 FA | 1,450-1,466 | 1,244-1,262 | 751-756 | base |
| + int8 prefill (`_C_dnnl.so` built against pip onednn-devel 2026.0.0, `EXL3_INT8_PREFILL=1`; kernel test 1.92x, worst 3.0e-4 vs torch ref) | 2,355 | 1,802 | 926 | yes |
| + oneDNN fused SDPA for long single-sequence fp8 prefill chunks (`EXL3_ONEDNN_ATTN=1`, `_patch_xpu_attn_routes`) | 2,342 | 1,970 | **1,404** | yes |
| + `--linear-attn-prefill-backend intel_xpu` (sgl_kernel fused GDN for plain extends) | 2,352 | **2,119** | 1,405 | yes (one sample) |
vLLM same card/day: 2,461 / 2,227 / 1,359. Ladder OK at every step, greedy output unchanged.

### GDN fold costs decode (A/B, same boot flags otherwise, C1/C4 prose thinking on, live log acceptance)
snapshot: C1 58.3 tok/s per stream @2.59 (44 ms/step), C4 44.5 @2.55 (57 ms/step); fold: C1 55.5 @2.98 (54 ms/step),
C4 32.2 @2.50 (78 ms/step). The fold commit (`gdn_replayssm_exact_fold_kernel`, num_warps=1) adds ~10-20 ms per step
on XPU. **Rejected as default** (speed); kept opt-in for the full-context-streams cell (KV 336-341K vs 257K).
Note: both arms ran after two PCIe drops and read ~15 % below this morning's sg4 at C1; cross-boot comparisons on
this card are no longer trustworthy.

### B70 #0 (84:00.0, slot 17) is not stable either: stopped
Slot 17 dropped off the bus at 12:01:18 (2 min into the sg17 panel) and 12:36:07 (sg18 panel, ~14 min of load),
after one drop yesterday (02:01, together with slot 19). Each re-enumerates the card (renderD135 -> 132 -> 134);
the 12:01 drop coincided with 71 RxErr lines on the RTX 3090 at c5:00.0 (another agent's card). Idle: no errors.
Panels sg17/sg18 are invalid past the drop points. GPU work stopped here.

### Live engine restore (2026-09-26 ~13:10 box time)
`docker start` of the original engine container failed: its device `/dev/dri/renderD135` no longer exists after the
drops. Restored through the plugin's own recovery path instead: `omarchy-local-ai stop` + `omarchy-local-ai run
qwen38-27b-exl3-4bpw-arcb70-vllm-exl3xpu-tp1 intel-xpu:0` (same recipe, same image
ghcr.io/0xsero/exl3xpu@sha256:86276b00, fresh by-path link). Controller state ready at 11:10:50 UTC; engine maps
renderD134 = 84:00.0; gateway `/v1/models` 200 and a chat request ("17*3" -> "51") through 127.0.0.1:12434;
deployment config keeps agent claude, folder ~/Work/la-typed-test. The original container record is in
`omarchy:~/sglb70/live-engine-inspect.json`.

### Replay redirect (Sero: "focus just on replaying real session JSONL") - prepared, not run
- Corpus: `~/tuning-kit/omp_replay_corpus.jsonl` (40 omp sessions) was NOT clean: redacted into
  `omarchy:~/sglb70/replay/omp_replay_corpus.redacted.jsonl` (`redact_corpus.py`, counts only: 1 Anthropic key,
  14 OpenAI-style keys, 1 AWS key id, 1 private key block, 13 bearer tokens, 109 password/token assignments;
  rescan 0 matches). The unredacted file never left the Mac.
- 262K filter (`bench/filter_replay_corpus.py`, chat template + tools, 16K output reserve, 12 turns): 40 kept,
  0 dropped; largest request 164,142 tokens (p50 25,420). -> `omp_replay_corpus.redacted.filtered.jsonl`.
- Harness `bench/replay_bench.py` (kit's cell_replay: same seed, tools, thinking on, t=1/top-p 0.95, no max_tokens):
  per-stream p50/p5, decode aggregate, output tok/s, TTFT p50/p95 split cached-prefix vs first turn, new-token
  prefill tok/s, turns/min, acceptance (vLLM /metrics or SGLang log), LOOP/REQ_FAIL/KV_FULL/TOOL_JSON, session wall
  time; ledger `omarchy:~/sglb70/ledger.jsonl`. Needs `--enable-prompt-tokens-details` (vLLM) / `--enable-cache-report`
  (SGLang) for server-reported cached tokens, else it estimates from the previous turn.
- Known SGLang gap for replay: prefix caching is OFF on the current config. `intel_xpu` attention needs page 64/128;
  `no_buffer` mamba radix needs page 1; `extra_buffer` is refused on XPU by `supports_mamba_cache_extra_buffer`
  (hard-coded False for XPU). First step on a stable card: lift that gate (as done for the fold) and validate.

## 2026-09-26 afternoon: "push" round on 84:00.0 (live engine stopped via `omarchy-local-ai stop` at ~13:40 box time)
- Power/clock lever: none without root. `tile0/gt0/freq0/max_freq`, `min_freq`, `power_profile` and hwmon
  `power1_cap` are root-owned (mode 644 / 664 root:root), no xpu-smi, and container sysfs is read-only; a privileged
  container would be root in all but name, so it was not used. Stock values recorded: max 2800 MHz, cap 230 W,
  profile base (both B70s). Drops cannot be A/B'd against power here: owner action needed.
- Resilience: `scripts/sglb70_supervise.py` (plan of cells, checkpoint per cell in ~/sglb70/ckpt, waits for the card
  to re-enumerate, relaunches the server on the new render node, retries the cell up to 3x, logs `kind=drop` rows to
  ~/sglb70/ledger.jsonl, aborts if any slot other than 17 goes link-down). Named configs in `scripts/cfg/*.args|env`.

### Prefix caching on XPU for the GDN hybrid: WORKS (`_patch_xpu_mamba_extra_buffer`, default on)
SGLang 0.5.20 hard-codes `supports_mamba_cache_extra_buffer() -> False` on XPU; `no_buffer` needs page 1, which
intel_xpu attention forbids. Lifting the platform gate (same arch + triton-GDN rule as CUDA) boots a Unified Radix
Cache (FULL + MAMBA components, page 128) with `--mamba-radix-cache-strategy extra_buffer`. `bench/pc_check.py`
(3 chained turns over one document, a different needle per turn, greedy, `--enable-cache-report`):
| doc | turn 1 TTFT (cold) | turn 2 / 3 TTFT | cached tokens | needles |
|---|---|---|---|---|
| 16.5K (2 seeds) | 7.2-8.8 s | 0.18-0.25 s | 16,512 | 6/6 |
| 65.8K | 36.0 s | 0.60 / 0.45 s | 65,664 / 65,792 | 3/3 |
Greedy identity: all 6 answers (2 seeds x 3 turns) byte-identical to the no-cache config (`nopc`). **Kept (cfg pc0).**
