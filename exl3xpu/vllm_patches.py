"""
Targeted fixes to vLLM internals that matter on XPU. Applied from the plugin entry point, each one
checks the exact source it expects and logs + skips if vLLM changed (never patches blindly).

gdn_mask_index: GDNAttentionMetadataBuilder.build indexes device tensors with a CPU boolean mask
  (e.g. block_table_tensor[spec_sequence_masks_cpu, :k]). That forces a pageable host->device copy of
  the mask, which on XPU's in-order queue waits for all prior GPU work: ~20 ms of stall every speculative
  decode step (measured: 1.0 s over 50 steps), serialising host prep with the previous step. The patch
  routes those through _rows(): a slice when the mask is all-true (pure spec decode, the hot path), else
  CPU-computed indices copied asynchronously from pinned memory + index_select (no sync).
"""
from __future__ import annotations
import copy
import inspect
import os
import re
import textwrap

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.exl3xpu")


def _rows(t: torch.Tensor, mask_cpu: torch.Tensor) -> torch.Tensor:
    """t[mask_cpu] without a host<->device sync (mask is a CPU bool tensor over t's first dim)."""
    if t.device.type == "cpu":
        return t[mask_cpu]
    n = mask_cpu.shape[0]
    if bool(mask_cpu.all()):
        return t[:n]
    idx = mask_cpu.nonzero().squeeze(1).pin_memory().to(t.device, non_blocking=True)
    return t.index_select(0, idx)


_PATTERNS = [
    # (regex on source, replacement)
    (r"(\w+)\[\s*spec_sequence_masks_cpu\s*,\s*: self\.num_spec \+ 1\s*\]", r"_rows(\1, spec_sequence_masks_cpu)[:, : self.num_spec + 1]"),
    (r"(\w+)\[\s*~spec_sequence_masks_cpu\s*,\s*0\s*\]", r"_rows(\1, ~spec_sequence_masks_cpu)[:, 0]"),
    (r"(\w+)\[\s*~spec_sequence_masks_cpu\s*\]", r"_rows(\1, ~spec_sequence_masks_cpu)"),
    (r"(\w+)\[\s*spec_sequence_masks_cpu\s*\]", r"_rows(\1, spec_sequence_masks_cpu)"),
]


def patch_gdn_mask_index() -> bool:
    try:
        from vllm.v1.attention.backends import gdn_attn
    except Exception as e:  # noqa
        logger.warning("exl3xpu: gdn_attn not importable (%s); GDN sync patch skipped", e)
        return False
    cls = getattr(gdn_attn, "GDNAttentionMetadataBuilder", None)
    if cls is None or getattr(cls, "_exl3_sync_patched", False):
        return False
    src = textwrap.dedent(inspect.getsource(cls.build))
    n_total = 0
    for pat, rep in _PATTERNS:
        src, n = re.subn(pat, rep, src)
        n_total += n
    if n_total < 2 or "_rows(block_table_tensor, spec_sequence_masks_cpu)" not in src:
        logger.warning("exl3xpu: GDNAttentionMetadataBuilder.build source changed (%d matches); sync patch skipped",
                       n_total)
        return False
    ns = dict(vars(gdn_attn))
    ns["_rows"] = _rows
    exec(compile(src, f"<exl3xpu patched {gdn_attn.__file__}>", "exec"), ns)
    cls.build = ns["build"]
    cls._exl3_sync_patched = True
    logger.info("exl3xpu: patched GDN metadata build (%d mask-index sites made sync-free)", n_total)
    return True


FP8KV_PREFILL_MIN_SEQ = 4096
FP8KV_PREFILL_MIN_QUERY = 64      # genuine prefill chunks only (not MTP verify batches of k+1 tokens)


def patch_fp8kv_prefill() -> bool:
    """Route single-sequence prefill chunks over an fp8 KV cache through block-dequantized fp16 attention
    (exl3xpu.fp8kv_prefill): the XPU FA2 kernel runs ~1.8x slower on fp8 K/V than on fp16."""
    try:
        from vllm.v1.attention.backends import flash_attn as fa_mod
    except Exception as e:  # noqa
        logger.warning("exl3xpu: flash_attn backend not importable (%s); fp8-KV prefill patch skipped", e)
        return False
    cls = getattr(fa_mod, "FlashAttentionImpl", None)
    if cls is None or getattr(cls, "_exl3_fp8kv_patched", False):
        return False
    from .fp8kv_prefill import prefill_attention
    orig = cls.forward
    is_q = fa_mod.is_quantized_kv_cache
    fp8_dtype = fa_mod.current_platform.fp8_dtype()
    decoder = fa_mod.AttentionType.DECODER

    def eligible(self, md, output_scale, output_block_scale) -> bool:
        if md is None or output_scale is not None or output_block_scale is not None:
            return False
        if not is_q(self.kv_cache_dtype) or getattr(md, "use_cascade", False):
            return False
        if md.max_query_len < FP8KV_PREFILL_MIN_QUERY or md.query_start_loc.shape[0] != 2 \
                or md.max_seq_len < FP8KV_PREFILL_MIN_SEQ:
            return False
        if torch.xpu.is_available() and torch.xpu.is_current_stream_capturing():
            return False
        if self.alibi_slopes is not None or getattr(self, "sinks", None) is not None:
            return False
        if self.logits_soft_cap not in (None, 0, 0.0) or getattr(self, "dcp_world_size", 1) != 1:
            return False
        if getattr(self, "attn_type", decoder) != decoder or not getattr(md, "causal", True):
            return False
        sw = getattr(md, "sliding_window", None) or self.sliding_window
        if sw is not None and tuple(sw) != (-1, -1):
            return False
        if getattr(md, "mm_prefix_range_tensor", None) is not None or getattr(md, "rswa_prefix_lens", None) is not None:
            return False
        return True

    def mixed_eligible(self, md, output_scale, output_block_scale) -> bool:
        """decode requests + exactly one long prefill chunk (decodes are ordered first by vLLM)"""
        if md is None or output_scale is not None or output_block_scale is not None:
            return False
        if os.environ.get("EXL3_FP8KV_MIXED", "0") != "1":
            return False
        if getattr(md, "num_prefill_reqs", 0) != 1 or getattr(md, "num_decode_reqs", 0) < 1:
            return False
        if md.num_actual_tokens - md.num_decode_tokens < FP8KV_PREFILL_MIN_QUERY or md.max_seq_len < FP8KV_PREFILL_MIN_SEQ:
            return False
        if not is_q(self.kv_cache_dtype) or getattr(md, "use_cascade", False):
            return False
        if torch.xpu.is_available() and torch.xpu.is_current_stream_capturing():
            return False
        if self.alibi_slopes is not None or getattr(self, "sinks", None) is not None:
            return False
        if self.logits_soft_cap not in (None, 0, 0.0) or getattr(self, "dcp_world_size", 1) != 1:
            return False
        if getattr(self, "attn_type", decoder) != decoder or not getattr(md, "causal", True):
            return False
        sw = getattr(md, "sliding_window", None) or self.sliding_window
        if sw is not None and tuple(sw) != (-1, -1):
            return False
        if getattr(md, "mm_prefix_range_tensor", None) is not None or getattr(md, "rswa_prefix_lens", None) is not None:
            return False
        return True

    def mixed_forward(self, layer, query, key, value, kv_cache, md, output):
        """decode rows through the stock kernel on sliced metadata; the one long prefill chunk through the
        block-dequantized fp16 path (a mixed step otherwise runs FA2 on fp8 K/V, ~1.8x slower, and every decode
        stream waits for it: the multi-second stalls under interleaved load)"""
        nd, ndt, n = md.num_decode_reqs, md.num_decode_tokens, md.num_actual_tokens
        if not getattr(cls, "_exl3_mixed_logged", False):
            cls._exl3_mixed_logged = True
            logger.info("exl3xpu: mixed fp8-KV step: %d decode reqs (%d tokens) + prefill chunk of %d tokens",
                        nd, ndt, n - ndt)
        dmd = copy.copy(md)
        dmd.num_actual_tokens = ndt
        dmd.query_start_loc = md.query_start_loc[: nd + 1]
        dmd.seq_lens = md.seq_lens[:nd]
        dmd.block_table = md.block_table[:nd]
        dmd.slot_mapping = md.slot_mapping[:ndt]
        dmd.max_query_len = ndt
        dmd.num_prefill_reqs, dmd.num_prefill_tokens = 0, 0
        if getattr(dmd, "scheduler_metadata", None) is not None:
            dmd.scheduler_metadata = None
        orig(self, layer, query[:ndt], key[:ndt] if key is not None else None, value[:ndt] if value is not None else None,
             kv_cache, dmd, output[:ndt])
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = key_cache.view(fp8_dtype)
        value_cache = value_cache.view(fp8_dtype)
        q = query[ndt:n]
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = (q.float() * float(layer._q_scale)).to(torch.float16)
        out = output[ndt:n].view(n - ndt, q.shape[1], self.head_size)
        tmp = out if out.dtype == torch.float16 else torch.empty(out.shape, dtype=torch.float16, device=out.device)
        ks = getattr(layer, "_k_scale_float", None)
        vs = getattr(layer, "_v_scale_float", None)
        ks = float(layer._k_scale) if ks is None else float(ks)
        vs = float(layer._v_scale) if vs is None else float(vs)
        seq_len = int(md.seq_lens[nd].item())
        prefill_attention(q.to(torch.float16), key_cache, value_cache, md.block_table[nd], seq_len,
                          ks, vs, float(self.scale), tmp)
        if tmp is not out:
            out.copy_(tmp)
        return output

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        md = attn_metadata
        if not eligible(self, md, output_scale, output_block_scale):
            if mixed_eligible(self, md, output_scale, output_block_scale):
                return mixed_forward(self, layer, query, key, value, kv_cache, md, output)
            return orig(self, layer, query, key, value, kv_cache, attn_metadata, output,
                        output_scale, output_block_scale)
        n = md.num_actual_tokens
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = key_cache.view(fp8_dtype)
        value_cache = value_cache.view(fp8_dtype)
        q = query[:n]
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = (q.float() * float(layer._q_scale)).to(torch.float16)
        out = output[:n].view(n, q.shape[1], self.head_size)
        tmp = out if out.dtype == torch.float16 else torch.empty(out.shape, dtype=torch.float16, device=out.device)
        # host-side scale copies (no device sync); fall back to the tensors only if vLLM lacks them
        ks = getattr(layer, "_k_scale_float", None)
        vs = getattr(layer, "_v_scale_float", None)
        ks = float(layer._k_scale) if ks is None else float(ks)
        vs = float(layer._v_scale) if vs is None else float(vs)
        prefill_attention(q.to(torch.float16), key_cache, value_cache, md.block_table[0], int(md.max_seq_len),
                          ks, vs, float(self.scale), tmp)
        if tmp is not out:
            out.copy_(tmp)
        return output

    cls.forward = forward
    cls._exl3_fp8kv_patched = True
    logger.info("exl3xpu: fp8-KV prefill attention routed through block-dequantized fp16 FA (>= %d tokens)",
                FP8KV_PREFILL_MIN_SEQ)
    return True


def patch_xpu_block_size() -> bool:
    """Opt-in (EXL3_KV_BLOCK_EXACT=1): keep the hybrid-negotiated attention block size (a multiple of 64 for the
    GDN kernel) instead of rounding it up to a power of two. On Qwen3.8 with fp8 KV the negotiation gives 1600
    tokens (= the 3.1 MiB GDN state page); the XPU platform rounds that to 2048 and pads every page to 4 MiB,
    wasting ~22% of the KV pool. The power-of-two rule exists for the ESIMD paged-attention decode fast path;
    this model's decode/verify attention runs through FA2 varlen instead."""
    import os
    if os.environ.get("EXL3_KV_BLOCK_EXACT") != "1":
        return False
    try:
        from vllm.platforms import xpu as xpu_mod
        from vllm.platforms.interface import Platform
    except Exception as e:  # noqa
        logger.warning("exl3xpu: XPU platform not importable (%s); block-size patch skipped", e)
        return False
    cls = getattr(xpu_mod, "XPUPlatform", None)
    if cls is None or getattr(cls, "_exl3_block_patched", False):
        return False

    def update_block_size_for_backend(klass, vllm_config) -> None:
        Platform.update_block_size_for_backend.__func__(klass, vllm_config)
        from vllm.config.vllm import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        cc = vllm_config.cache_config
        layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
        if not any(l.get_attn_backend().get_name() == "GDN_ATTN" for l in layers.values()):
            return
        bs = max(cc.block_size, 64)
        new = (bs + 63) // 64 * 64
        if new == cc.block_size:
            logger.info("exl3xpu: keeping attention block size %d (multiple of 64, not rounded to a power of 2)", new)
            return
        if cc.mamba_cache_mode == "align":
            cc.mamba_block_size = new
        if cc.mamba_page_size_padded is not None:
            cc.mamba_page_size_padded = new * (cc.mamba_page_size_padded // cc.block_size)
        cc.block_size = new
        logger.info("exl3xpu: attention block size %d (multiple of 64)", new)

    cls.update_block_size_for_backend = classmethod(update_block_size_for_backend)
    cls._exl3_block_patched = True
    logger.info("exl3xpu: XPU block size keeps the hybrid negotiation (EXL3_KV_BLOCK_EXACT=1)")
    return True


def apply_all():
    patch_gdn_mask_index()
    patch_fp8kv_prefill()
    patch_xpu_block_size()
