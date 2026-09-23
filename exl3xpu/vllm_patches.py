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
import inspect
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


def apply_all():
    patch_gdn_mask_index()
