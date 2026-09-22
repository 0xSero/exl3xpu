"""
vLLM out-of-tree quantization plugin: EXL3 (exllamav3 trellis) weights on Intel XPU.

Registered via the `vllm.general_plugins` entry point, so every vLLM process (API server,
engine core, workers) imports it before model construction.
"""
from __future__ import annotations
import json
import os
import re
from typing import Any

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

logger = init_logger(__name__)

# vLLM fused module -> checkpoint constituents
FUSED = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}
QKV_IDS = {"q": 0, "k": 1, "v": 2}
CB_IDS = {"3inst": 0, "mcg": 1, "mul1": 2}


def _norm_key(k: str) -> str:
    """'model.language_model.layers.3.mlp.gate_proj' / 'language_model.model.layers.3...' -> 'layers.3.mlp.gate_proj'"""
    m = re.search(r"(layers\.\d+\..*)$", k)
    if m:
        return m.group(1)
    return k.split(".")[-1]  # lm_head, etc.


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):

    def __init__(self, bits: float, head_bits: int, codebook: str, storage: dict | None = None):
        super().__init__()
        self.bits = bits
        self.head_bits = head_bits
        self.codebook = codebook
        self.storage = storage or {}

    def __repr__(self):
        return f"Exl3Config(bits={self.bits}, head_bits={self.head_bits}, codebook={self.codebook})"

    def get_name(self):
        return "exl3"

    def get_supported_act_dtypes(self):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(bits=config.get("bits", 4), head_bits=config.get("head_bits", 6),
                   codebook=config.get("codebook", "mul1"))

    def maybe_update_config(self, model_name: str, hf_config=None, revision=None):
        path = os.path.join(model_name, "quantization_config.json")
        if not os.path.isfile(path):
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(model_name, "quantization_config.json", revision=revision)
            except Exception:
                logger.warning("exl3: no quantization_config.json found; assuming all linears are EXL3")
                return
        with open(path) as f:
            ts = json.load(f).get("tensor_storage", {})
        for key, v in ts.items():
            if v.get("quant_format") == "exl3":
                self.storage[_norm_key(key)] = int(v.get("bits_per_weight", self.bits))
        logger.info("exl3: %d quantized tensors in storage map", len(self.storage))

    def _bits_for(self, prefix: str) -> int | None:
        key = _norm_key(prefix)
        if not self.storage:
            if key.endswith("in_proj_ba"):
                return None
            return self.head_bits if key == "lm_head" else int(self.bits)
        base, _, leaf = key.rpartition(".")
        parts = FUSED.get(leaf, [leaf])
        names = [f"{base}.{p}" if base else p for p in parts]
        bits = [self.storage.get(n) for n in names]
        if all(b is None for b in bits):
            return None
        assert all(b == bits[0] for b in bits), f"exl3: mixed quant in fused module {prefix}: {bits}"
        return bits[0]

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            bits = self._bits_for(prefix)
            if bits is None:
                return UnquantizedLinearMethod()
            return Exl3LinearMethod(self, bits)
        if isinstance(layer, ParallelLMHead):
            bits = self._bits_for(prefix)
            if bits is None:
                return None
            return Exl3LinearMethod(self, bits)
        return None


class Exl3LinearMethod(LinearMethodBase):

    def __init__(self, config: Exl3Config, bits: int):
        self.config = config
        self.K = int(bits)
        self.cb = CB_IDS[config.codebook]

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size,
                       output_size, params_dtype, **extra_weight_attrs):
        k = input_size_per_partition
        n = sum(output_partition_sizes)
        if k != input_size or n != (output_size if not isinstance(layer, ParallelLMHead) else n):
            raise NotImplementedError("exl3: tensor parallel sharding not supported yet (use DP)")
        assert k % 128 == 0 and all(s % 128 == 0 for s in output_partition_sizes), \
            f"exl3: dims must be multiples of 128 (k={k}, n={output_partition_sizes})"
        K = self.K
        layer.exl3_sizes = list(output_partition_sizes)
        layer.exl3_offsets = [sum(output_partition_sizes[:i]) for i in range(len(output_partition_sizes) + 1)]
        layer.exl3_groups = set()
        layer.exl3_K = K
        layer.exl3_cb = self.cb

        def reg(name, tensor, loader):
            p = Parameter(tensor, requires_grad=False)
            p.weight_loader = loader
            layer.register_parameter(name, p)

        reg("trellis", torch.empty((k // 16, n // 16, 16 * K), dtype=torch.int16), self._load_trellis(layer))
        reg("suh", torch.empty((len(output_partition_sizes), k), dtype=torch.float16), self._load_suh(layer))
        reg("svh", torch.empty((n,), dtype=torch.float16), self._load_svh(layer))
        if self.config.codebook != "3inst":
            reg(self.config.codebook, torch.empty((), dtype=torch.int32), lambda p, w, *a, **kw: p.data.copy_(w))

    @staticmethod
    def _shards(layer, loaded, sid=None) -> list[int]:
        if sid is None:
            sid = getattr(loaded, "shard_id", None)
        if sid is None:
            return list(range(len(layer.exl3_sizes)))
        if isinstance(sid, str):
            return [QKV_IDS[sid]]
        if isinstance(sid, int):
            return [sid]
        return list(sid)

    def _range(self, layer, loaded, sid=None):
        s = self._shards(layer, loaded, sid)
        assert s == list(range(s[0], s[-1] + 1)), s
        return s, layer.exl3_offsets[s[0]], layer.exl3_offsets[s[-1] + 1]

    def _load_trellis(self, layer):
        def f(param, w, sid=None):
            s, a, b = self._range(layer, w, sid)
            assert w.dtype == torch.int16 and w.shape[-1] == param.shape[-1], \
                f"exl3: trellis bits mismatch {tuple(w.shape)} vs {tuple(param.shape)}"
            assert w.shape[1] * 16 == b - a and w.shape[0] == param.shape[0], (w.shape, a, b)
            param.data[:, a // 16: b // 16].copy_(w)
            layer.exl3_groups.add(tuple(s))
        return f

    def _load_suh(self, layer):
        def f(param, w, sid=None):
            s, _, _ = self._range(layer, w, sid)
            for i in s:
                param.data[i].copy_(w)
        return f

    def _load_svh(self, layer):
        def f(param, w, sid=None):
            _, a, b = self._range(layer, w, sid)
            param.data[a:b].copy_(w)
        return f

    def process_weights_after_loading(self, layer):
        groups = sorted(layer.exl3_groups) or [tuple(range(len(layer.exl3_sizes)))]
        covered = [i for g in groups for i in g]
        assert covered == list(range(len(layer.exl3_sizes))), f"exl3: incomplete shards {groups}"
        dev = layer.trellis.device
        suh = torch.stack([layer.suh.data[g[0]] for g in groups]).contiguous()
        bounds = [layer.exl3_offsets[g[0]] for g in groups] + [layer.exl3_offsets[-1]]
        n = bounds[-1]
        shard_of_nb = torch.empty(n // 128, dtype=torch.int32)
        for gi in range(len(groups)):
            shard_of_nb[bounds[gi] // 128: bounds[gi + 1] // 128] = gi
        layer.suh = Parameter(suh, requires_grad=False)
        layer.exl3_shard_of_nb = shard_of_nb.to(dev)
        layer.exl3_bounds = bounds
        cb = self.cb
        if hasattr(layer, "mcg"):
            cb = 1
        elif hasattr(layer, "mul1"):
            cb = 2
        layer.exl3_cb = cb

    def apply(self, layer, x, bias=None):
        from .ops import exl3_linear
        y = exl3_linear(x, layer.trellis, layer.suh, layer.svh, layer.exl3_shard_of_nb,
                        layer.exl3_bounds, layer.exl3_K, layer.exl3_cb)
        if bias is not None:
            y = y + bias
        return y

    # lm_head (ParallelLMHead) path
    def embedding(self, layer, input_):
        raise NotImplementedError("exl3: quantized input embeddings are not supported")


def register():
    """vllm.general_plugins entry point."""
    # import side effect registers the config
    return None
