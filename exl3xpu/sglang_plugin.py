"""
SGLang out-of-tree quantization plugin: EXL3 (exllamav3 trellis) weights on Intel XPU (Battlemage).

Registered through SGLang's `sglang.srt.plugins` entry point (runs in the launcher, the tokenizer/detokenizer and
every scheduler process), so stock SGLang serves `--quantization exl3` with no source edits. The kernels are the
same ESIMD/XMX op library the vLLM plugin uses (`torch.ops.exl3xpu_C.linear`); only the engine glue differs.

Storage per SGLang fused linear (qkv_proj, gate_up_proj, in_proj_qkvz, ...): each checkpoint matrix is one "group"
with its own input scale vector `suh`; trellis tiles are concatenated along n in partition order, `svh` likewise,
and `shard_of_nb` maps every 128-column output block to its group (exactly the vLLM plugin's layout).
"""
from __future__ import annotations

import json
import logging
import os
import re
import struct
from typing import Any

import torch

logger = logging.getLogger(__name__)
_done = False

_SUFFIXES = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1")
_QKV = {"q": 0, "k": 1, "v": 2}
_CB = {"3inst": 0, "mcg": 1, "mul1": 2}
_PACKED = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}
TARGET_LM_HEADS: list = []


def _dev() -> torch.device:
    return torch.device("xpu", torch.xpu.current_device())


def _empty_cache():
    try:
        torch.xpu.empty_cache()
    except Exception:
        pass


def norm_key(k: str) -> str:
    """Canonical module name shared by checkpoint keys and SGLang prefixes:
    'model.language_model.layers.3.mlp.gate_proj' / 'model.layers.3.mlp.gate_proj' -> 'layers.3.mlp.gate_proj';
    'mtp.layers.0.mlp.gate_proj' -> 'mtp.layers.0.mlp.gate_proj'; '...mtp.fc' -> 'mtp.fc'; '...lm_head' -> 'lm_head'."""
    parts = k.split(".")
    if "mtp" in parts:
        return "mtp." + ".".join(parts[parts.index("mtp") + 1:])
    m = re.search(r"(layers\.\d+\..*)$", k)
    if m:
        return m.group(1)
    return parts[-1]


def _read_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def scan_checkpoint(model_dir: str) -> dict[str, tuple[int, int, int, int]]:
    """{canonical module: (k, n, K bits, codebook id)} from the safetensors headers (never from JSON hints)."""
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.isfile(idx) else \
        [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    tensors: dict[str, list] = {}
    for fn in files:
        for name, meta in _read_header(os.path.join(model_dir, fn)).items():
            if name != "__metadata__":
                tensors[name] = meta["shape"]
    out = {}
    for name, shape in tensors.items():
        if not name.endswith(".trellis"):
            continue
        base = name[: -len(".trellis")]
        cb = "mcg" if base + ".mcg" in tensors else "mul1" if base + ".mul1" in tensors else "3inst"
        out[norm_key(base)] = (shape[0] * 16, shape[1] * 16, shape[2] // 16, _CB[cb])
    return out


def _partitions(sid) -> tuple[int, ...]:
    if sid is None:
        return ()
    if isinstance(sid, (tuple, list)):
        return tuple(sid)
    return (_QKV.get(sid, sid),)


def _unpack_signs(packed: torch.Tensor) -> torch.Tensor:
    bits = (packed.to(torch.int32).unsqueeze(1) >> torch.arange(16, device=packed.device)) & 1
    return (1.0 - 2.0 * bits.flatten()).to(torch.float16)


def _build_classes():
    from sglang.srt.layers.quantization.base_config import LinearMethodBase, QuantizationConfig
    from sglang.srt.utils.common import set_weight_attrs

    class Exl3XpuConfig(QuantizationConfig):
        def __init__(self, declared: dict | None = None, model_path: str | None = None):
            super().__init__()
            self.declared = {k: v for k, v in (declared or {}).items() if k not in ("hf_config", "packed_modules_mapping")}
            self.model_path = model_path
            self.modules = scan_checkpoint(model_path) if model_path else {}
            if model_path:
                bits = {}
                for v in self.modules.values():
                    bits[v[2]] = bits.get(v[2], 0) + 1
                logger.info("exl3xpu: %s: %d EXL3 linears, by bits %s, %d in the MTP head", model_path,
                            len(self.modules), bits, sum(1 for k in self.modules if k.startswith("mtp.")))

        def get_name(self) -> str:
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
        def from_config(cls, config: dict[str, Any]):
            hf_config = config.get("hf_config")
            hf_path = getattr(hf_config, "_name_or_path", None)
            path = os.environ.get("EXL3_MODEL_PATH") or hf_path
            if path and not os.path.isdir(path):
                from huggingface_hub import snapshot_download
                path = snapshot_download(path, local_files_only=True)
            if not path:
                raise ValueError("exl3xpu: model path unknown (set EXL3_MODEL_PATH)")
            cfg = cls(config, path)
            if config.get("packed_modules_mapping"):
                cfg.packed_modules_mapping = dict(config["packed_modules_mapping"])
            return cfg

        @classmethod
        def override_quantization_method(cls, hf_quant_cfg, user_quant):
            if isinstance(hf_quant_cfg, dict) and hf_quant_cfg.get("quant_method") == "exl3" and user_quant in (None, "exl3"):
                return "exl3"
            return None

        def get_scaled_act_names(self):
            return []

        def __getstate__(self):
            return self.__dict__.copy()

        def lookup(self, prefix: str, draft: bool = False):
            key = norm_key(prefix)
            if draft and not key.startswith("mtp.") and key.startswith("layers."):
                key = "mtp." + key
            return self.modules.get(key)

        def _sources(self, prefix: str) -> list[str]:
            parent, _, leaf = prefix.rpartition(".")
            packed = (getattr(self, "packed_modules_mapping", None) or _PACKED).get(leaf) or _PACKED.get(leaf)
            return [f"{parent}.{s}" if parent else s for s in packed] if packed else [prefix]

        def get_quant_method(self, layer: torch.nn.Module, prefix: str):
            from sglang.srt.layers.linear import LinearBase
            from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
            from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
            parts = prefix.split(".")
            if "visual" in parts or "vision_tower" in parts:
                return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else None
            draft = prefix.startswith("mtp") or ".mtp." in prefix or getattr(layer, "_exl3_draft", False)
            if isinstance(layer, ParallelLMHead):
                info = self.lookup(prefix, draft)
                if info is None and not draft:
                    info = self.modules.get("lm_head")
                return Exl3XpuLinearMethod(prefix, [info]) if info else None
            if not isinstance(layer, LinearBase):
                return None
            infos = [self.lookup(p, draft) for p in self._sources(prefix)]
            if not any(infos):
                return UnquantizedLinearMethod()
            if not all(infos):
                raise ValueError(f"exl3xpu: {prefix} mixes EXL3 and unquantized sources {self._sources(prefix)}")
            return Exl3XpuLinearMethod(prefix, infos)

    class Exl3XpuLinearMethod(LinearMethodBase):
        def __init__(self, prefix: str, infos: list[tuple]):
            self.prefix = prefix
            self.infos = infos       # per checkpoint matrix: (k, n, K, cb)

        def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                           params_dtype, **extra):
            if input_size_per_partition != input_size or sum(output_partition_sizes) != output_size:
                raise NotImplementedError(f"exl3xpu: {self.prefix}: tensor parallel EXL3 is not supported (use DP)")
            layer.exl3_out_sizes = list(output_partition_sizes)
            layer.exl3_in_size = input_size
            layer.exl3_shards = {s: {} for s in _SUFFIXES}
            for suffix in _SUFFIXES:
                p = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
                set_weight_attrs(p, {"weight_loader": self._loader(layer, suffix), "exl3_placeholder": True})
                layer.register_parameter(suffix, p)
            if not hasattr(layer, "weight"):
                # ParallelLMHead: code paths read `.weight` (shape/dtype); zero-width placeholder
                w = torch.nn.Parameter(torch.empty((sum(output_partition_sizes), 0), dtype=params_dtype),
                                       requires_grad=False)
                set_weight_attrs(w, {"weight_loader": lambda *a, **k: None, "exl3_placeholder": True})
                layer.register_parameter("weight", w)

        def _loader(self, layer, suffix):
            def load(param, loaded_weight, loaded_shard_id=None, *a, **k):
                store = layer.exl3_shards[suffix]
                key = tuple(loaded_shard_id) if isinstance(loaded_shard_id, list) else loaded_shard_id
                if key in store:
                    raise ValueError(f"exl3xpu: {self.prefix}.{suffix}: shard {key!r} loaded twice")
                store[key] = loaded_weight.to(_dev(), copy=True)
            return load

        def process_weights_after_loading(self, layer) -> None:
            if not hasattr(layer, "exl3_shards"):
                return
            got = layer.exl3_shards
            ids = sorted(got["trellis"], key=lambda sid: _partitions(sid) or (0,))
            if not ids:
                logger.info("exl3xpu: %s: no EXL3 tensors delivered (shared from the target?)", self.prefix)
                layer.exl3_empty = True
                return
            if len(ids) != len(self.infos):
                raise ValueError(f"exl3xpu: {self.prefix}: expected {len(self.infos)} matrices, got {len(ids)} ({ids})")
            trellis, suh, svh, bounds, Ks, cbs = [], [], [], [0], set(), set()
            for i, sid in enumerate(ids):
                su, sv = got["suh"].get(sid), got["svh"].get(sid)
                if su is None and sid in got["su"]:
                    su, sv = _unpack_signs(got["su"][sid]), _unpack_signs(got["sv"][sid])
                if su is None or sv is None:
                    raise ValueError(f"exl3xpu: {self.prefix}: shard {sid!r} has no scale vectors")
                t = got["trellis"][sid]
                cb = 1 if sid in got["mcg"] else 2 if sid in got["mul1"] else 0
                k, n, K, cb_decl = self.infos[i]
                if (t.shape[0] * 16, t.shape[1] * 16, t.shape[2] // 16, cb) != (k, n, K, cb_decl):
                    raise ValueError(f"exl3xpu: {self.prefix}: shard {sid!r} {tuple(t.shape)} cb={cb} != header {(k, n, K, cb_decl)}")
                parts = _partitions(sid)
                width = sum(layer.exl3_out_sizes[p] for p in parts) if parts else sum(layer.exl3_out_sizes)
                if width != n:
                    raise ValueError(f"exl3xpu: {self.prefix}: shard {sid!r} is {n} wide, SGLang expects {width}")
                trellis.append(t); suh.append(su.to(torch.float16)); svh.append(sv.to(torch.float16))
                bounds.append(bounds[-1] + n); Ks.add(K); cbs.add(cb)
            if len(Ks) != 1 or len(cbs) != 1:
                raise ValueError(f"exl3xpu: {self.prefix}: fused group mixes bitrates/codebooks {Ks} {cbs}")
            if bounds[-1] % 128 or layer.exl3_in_size % 128:
                raise ValueError(f"exl3xpu: {self.prefix}: dims must be multiples of 128 ({layer.exl3_in_size}, {bounds})")
            for suffix in _SUFFIXES:
                delattr(layer, suffix)
            del layer.exl3_shards
            dev = trellis[0].device
            layer.register_buffer("exl3_trellis", torch.cat(trellis, 1).contiguous() if len(trellis) > 1 else trellis[0].contiguous(), persistent=False)
            layer.register_buffer("exl3_suh", torch.stack(suh).contiguous(), persistent=False)
            layer.register_buffer("exl3_svh", torch.cat(svh).contiguous(), persistent=False)
            sonb = torch.empty(bounds[-1] // 128, dtype=torch.int32)
            for g in range(len(bounds) - 1):
                sonb[bounds[g] // 128: bounds[g + 1] // 128] = g
            layer.register_buffer("exl3_shard_of_nb", sonb.to(dev), persistent=False)
            layer.exl3_bounds = bounds
            layer.exl3_K = Ks.pop()
            layer.exl3_cb = cbs.pop()
            del trellis
            _empty_cache()
            from . import ops
            E = ops._get_esimd()
            if not (E and hasattr(E, "linear") and E.exl3_supported(layer.exl3_K, layer.exl3_cb)):
                raise RuntimeError(f"exl3xpu: {self.prefix}: K={layer.exl3_K} cb={layer.exl3_cb} not built into _C.so "
                                   "(rebuild with EXL3_FLAGS=-DEXL3_ALL_CODEBOOKS)")
            if self.prefix.endswith("lm_head") and not self.prefix.startswith("mtp"):
                TARGET_LM_HEADS.append(layer)
                if os.environ.get("EXL3_DRAFT_VOCAB"):
                    _build_draft_head(layer, os.environ["EXL3_DRAFT_VOCAB"])

        def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
            from . import ops
            y = torch.ops.exl3xpu_C.linear(x, layer.exl3_trellis, layer.exl3_suh, layer.exl3_svh,
                                           layer.exl3_shard_of_nb, layer.exl3_bounds, layer.exl3_K, layer.exl3_cb,
                                           ops.SMALL_M_MAX, ops.RECON_SLICE_N)
            return y if bias is None else y + bias

        def embedding(self, layer, input_):
            raise NotImplementedError("exl3xpu: quantized input embeddings are not supported")

    class Exl3Dense(torch.nn.Module):
        """Drop-in for a plain nn.Linear whose checkpoint tensors are EXL3 (the MTP head's `fc`)."""

        def __init__(self, in_features: int, out_features: int, info: tuple, prefix: str, params_dtype=torch.float16):
            super().__init__()
            self.in_features, self.out_features = in_features, out_features
            self.prefix = prefix
            self.method = Exl3XpuLinearMethod(prefix, [info])
            self.method.create_weights(self, in_features, [out_features], in_features, out_features, params_dtype)

        def process_weights_after_loading(self):
            self.method.process_weights_after_loading(self)

        def forward(self, x):
            return self.method.apply(self, x)

    return Exl3XpuConfig, Exl3XpuLinearMethod, Exl3Dense


_CLASSES = None


def classes():
    global _CLASSES
    if _CLASSES is None:
        _CLASSES = _build_classes()
    return _CLASSES


# ---------------------------------------------------------------------------------------------------------------
# Pruned-vocabulary MTP draft head (same idea as the vLLM plugin): the drafter proposes from a subset of 128-token
# lm_head blocks; the target verifies with the full head, so outputs are unchanged. EXL3_DRAFT_VOCAB=<json>.

def _build_draft_head(layer, path):
    with open(path) as f:
        spec = json.load(f)
    blocks = torch.tensor(spec["blocks"], dtype=torch.long)
    nb = layer.exl3_svh.shape[0] // 128
    blocks = blocks[blocks < nb]
    dev = layer.exl3_trellis.device
    tiles = (blocks[:, None] * 8 + torch.arange(8)[None, :]).flatten().to(dev)
    layer.exl3_draft = dict(
        trellis=layer.exl3_trellis.index_select(1, tiles).contiguous(),
        svh=layer.exl3_svh.view(-1, 128).index_select(0, blocks.to(dev)).flatten().contiguous(),
        idx=(blocks[:, None] * 128 + torch.arange(128)[None, :]).flatten().to(dev),
        shard=torch.zeros(len(blocks), dtype=torch.int32, device=dev),
        bounds=[0, len(blocks) * 128])
    logger.info("exl3xpu: pruned draft head: %d of %d vocab blocks", len(blocks), nb)


def _patch_mtp() -> None:
    """Qwen3.5 MTP draft: `fc` is a plain nn.Linear upstream but EXL3 checkpoints quantize `mtp.fc`; the draft's own
    embed_tokens is replaced by the target's (share it from construction so its bf16 copy never occupies VRAM)."""
    try:
        from sglang.srt.models import qwen3_5_mtp as m
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: MTP shim not installed (%s)", e)
        return
    cls = m.Qwen3_5ForCausalLMMTP
    if getattr(cls, "_exl3_patched", False):
        return
    orig_init, orig_load = cls.__init__, cls.load_weights

    def __init__(self, config, quant_config=None, prefix="", *a, **k):
        orig_init(self, config, quant_config, prefix, *a, **k)
        Cfg, _, Dense = classes()
        qc = getattr(self, "quant_config", quant_config)
        if not isinstance(qc, Cfg):
            return
        emb = self.model.embed_tokens
        w = emb.weight
        emb.weight = torch.nn.Parameter(torch.empty((0, w.shape[1]), dtype=w.dtype, device=w.device), requires_grad=False)
        for name, val in vars(w).items():
            if name != "data" and not hasattr(emb.weight, name):
                setattr(emb.weight, name, val)
        self._exl3_skip_embed = True
        del w
        _empty_cache()
        info = qc.modules.get("mtp.fc")
        if info is not None and isinstance(getattr(self, "fc", None), torch.nn.Linear):
            self.fc = Dense(self.fc.in_features, self.fc.out_features, info, "mtp.fc", params_dtype=self.fc.weight.dtype)
            logger.info("exl3xpu: MTP fc served as an EXL3 linear %s", info)

    def load_weights(self, weights, *a, **k):
        if getattr(self, "_exl3_skip_embed", False):
            weights = ((n, w) for n, w in weights if not n.endswith("embed_tokens.weight"))
        out = orig_load(self, weights, *a, **k)
        fc = getattr(self, "fc", None)
        if fc is not None and hasattr(fc, "exl3_shards"):
            fc.process_weights_after_loading()
        return out

    cls.__init__, cls.load_weights, cls._exl3_patched = __init__, load_weights, True


def _sample_rows(logits: torch.Tensor, temps: torch.Tensor, top_ks: torch.Tensor | None,
                 top_ps: torch.Tensor | None) -> torch.Tensor:
    """One token per row from softmax(logits / T) restricted to top-k and top-p (torch only, any device)."""
    probs = torch.softmax(logits.float() / temps.float().view(-1, 1), dim=-1)
    sp, si = probs.sort(dim=-1, descending=True)
    keep = torch.ones_like(sp, dtype=torch.bool)
    if top_ks is not None:
        ranks = torch.arange(sp.shape[1], device=sp.device).view(1, -1)
        keep &= ranks < top_ks.view(-1, 1).clamp_min(1)
    if top_ps is not None:
        keep &= (sp.cumsum(-1) - sp) < top_ps.float().view(-1, 1)
    keep[:, 0] = True
    sp = sp * keep
    pick = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1)
    return si.gather(1, pick)


def _patch_xpu_spec_sampling() -> None:
    """SGLang 0.5.20 on XPU verifies speculative drafts greedily whatever the request's temperature (eagle_sample:
    `is_all_greedy or ... or _is_xpu`), i.e. sampling is silently turned off whenever MTP is on. Restore it: sample one
    target token per verify row from the request's distribution (temperature, top-k, top-p) and make it the row's
    argmax, so the stock greedy chain verify accepts a draft token iff it equals the sampled target token. For a
    deterministic top-1 draft chain this is lossless (every emitted token is a sample of the target distribution).
    Penalties/logit bias are applied by eagle_sample after this and are not reflected in the sample (not used here).
    EXL3_SGL_SPEC_SAMPLE=0 restores the stock behaviour."""
    if os.environ.get("EXL3_SGL_SPEC_SAMPLE", "1") != "1":
        return
    try:
        from sglang.srt.speculative import eagle_worker_common as ewc
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: spec sampling patch not installed (%s)", e)
        return
    if getattr(ewc, "_exl3_sample_patched", False) or not hasattr(ewc, "eagle_sample"):
        return
    orig = ewc.eagle_sample

    def eagle_sample(verify_input, batch, logits_output, *a, **k):
        si = getattr(batch, "sampling_info", None)
        logits = getattr(logits_output, "next_token_logits", None)
        if (si is not None and logits is not None and logits.device.type == "xpu" and not si.is_all_greedy
                and not batch.forward_mode.is_idle()):
            n = verify_input.draft_token_num
            rep = lambda t: None if t is None else torch.repeat_interleave(t.view(-1), n, dim=0)
            tok = _sample_rows(logits, rep(si.temperatures),
                               rep(si.top_ks) if si.need_top_k_sampling else None,
                               rep(si.top_ps) if si.need_top_p_sampling else None)
            logits.scatter_(1, tok, torch.finfo(logits.dtype).max)
        return orig(verify_input, batch, logits_output, *a, **k)

    ewc.eagle_sample, ewc._exl3_sample_patched = eagle_sample, True
    logger.info("exl3xpu: XPU speculative verify samples the target distribution (was greedy-only)")


def _patch_xpu_gdn_verify() -> None:
    """SGLang 0.5.20's XPU copy of the fused sigmoid-gating delta-rule wrapper
    (hardware_backend/xpu/kernels/fla) is stale against the shared Triton kernel: it omits `stride_h0_source`
    (TypeError on the first MTP verify step) and the per-request pitch of the intermediate-state buffer. The generic
    wrapper in sglang.kernels.ops already launches the non-CUDA grid, so route the Triton GDN kernel class to it.
    EXL3_SGL_GDN_FIX=0 disables."""
    if os.environ.get("EXL3_SGL_GDN_FIX", "1") != "1":
        return
    try:
        if not torch.xpu.is_available():
            return
        from sglang.srt.layers.attention.linear.kernels import gdn_triton
        from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update as generic)
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: GDN verify fix not installed (%s)", e)
        return
    gdn_triton.fused_sigmoid_gating_delta_rule_update = generic
    logger.info("exl3xpu: XPU GDN verify routed to the generic fused_sigmoid_gating_delta_rule_update wrapper")


def _allow_xpu_in(module, names: list[str]) -> list[str]:
    """Re-define `module.<name>` from its source with every `<t>.is_cuda` test accepting XPU tensors as well
    (SGLang guards some device-agnostic Triton launchers with CUDA-only checks)."""
    import inspect
    import textwrap
    done = []
    for name in names:
        fn = getattr(module, name, None)
        if fn is None or getattr(fn, "_exl3_xpu_ok", False):
            continue
        src = textwrap.dedent(inspect.getsource(fn))
        new = re.sub(r"(\b[A-Za-z_][A-Za-z0-9_]*)\.is_cuda\b", r'(\1.device.type in ("cuda", "xpu"))', src)
        if new == src:
            continue
        ns: dict = {}
        exec(compile(new, f"<exl3xpu:{module.__name__}.{name}>", "exec"), module.__dict__, ns)
        ns[name]._exl3_xpu_ok = True
        setattr(module, name, ns[name])
        done.append(name)
    return done


def _patch_xpu_mamba_scatter() -> None:
    """MTP verify on hybrid GDN models commits the accepted step's recurrent/conv state with Triton scatter kernels
    whose Python launchers refuse non-CUDA tensors. EXL3_SGL_SCATTER_FIX=0 disables."""
    if os.environ.get("EXL3_SGL_SCATTER_FIX", "1") != "1":
        return
    try:
        if not torch.xpu.is_available():
            return
        from sglang.kernels.ops.mamba import mamba_state_scatter_triton as m
        done = _allow_xpu_in(m, ["fused_mamba_state_scatter_with_mask", "fused_conv_window_scatter_with_mask"])
        logger.info("exl3xpu: XPU allowed in mamba state scatter launchers %s", done)
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: mamba scatter XPU fix not installed (%s)", e)


def activate() -> None:
    """sglang.srt.plugins entry point."""
    global _done
    if _done:
        return
    _done = True
    try:
        from sglang.srt.layers.quantization import QUANTIZATION_METHODS
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: SGLang not importable (%s)", e)
        return
    Cfg, _, _ = classes()
    QUANTIZATION_METHODS["exl3"] = Cfg
    try:
        from sglang.srt.arg_groups.choices import QUANTIZATION_CHOICES, add_quantization_method_choices
        if "exl3" not in QUANTIZATION_CHOICES:
            add_quantization_method_choices(["exl3"])
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: could not add 'exl3' to --quantization choices (%s)", e)
    _patch_mtp()
    _patch_xpu_spec_sampling()
    _patch_xpu_gdn_verify()
    _patch_xpu_mamba_scatter()
    logger.info("exl3xpu: registered SGLang quantization method 'exl3' (XPU)")
