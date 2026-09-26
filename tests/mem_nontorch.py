"""Which first-use paths take device memory outside the torch allocator (Level Zero / UR pools, kernel modules)?"""
import torch, os
g = 2 ** 30
def rep(tag):
    torch.xpu.synchronize(); f, t = torch.xpu.mem_get_info(); r = torch.xpu.memory_reserved()
    print(f"{tag:34s} free {f/g:6.2f}  torch reserved {r/g:6.2f}  non-torch {(t-f-r)/g:6.2f} GiB", flush=True)
rep("start")
x = torch.ones(1024, device="xpu"); rep("tiny tensor op")
a = torch.randn(4096, 5120, device="xpu", dtype=torch.bfloat16); b = torch.randn(5120, 5120, device="xpu", dtype=torch.bfloat16)
c = a @ b; rep("bf16 matmul (oneDNN)")
from exl3xpu import ops
E = ops._get_esimd()
k, n = 5120, 17408
tr = torch.randint(-32768, 32767, (k // 16, n // 16, 64), dtype=torch.int16, device="xpu")
suh = torch.ones(1, k, device="xpu", dtype=torch.float16); svh = torch.ones(n, device="xpu", dtype=torch.float16)
sonb = torch.zeros(n // 128, dtype=torch.int32, device="xpu")
for M in (1, 4, 32, 4096):
    x = torch.randn(M, k, device="xpu", dtype=torch.bfloat16)
    y = E.linear(x, tr, suh, svh, sonb, [0, n], 4, 2, ops.SMALL_M_MAX, ops.RECON_SLICE_N); rep(f"exl3 linear M={M}")
from sgl_kernel.flash_attn import flash_attn_with_kvcache
kc = torch.zeros(64, 128, 4, 256, device="xpu", dtype=torch.float8_e4m3fn); vc = torch.zeros_like(kc)
q = torch.randn(4096, 24, 256, device="xpu", dtype=torch.bfloat16); one = torch.ones((), device="xpu").expand(1, 4)
o = flash_attn_with_kvcache(q=q, k_cache=kc, v_cache=vc, page_table=torch.arange(40, device="xpu", dtype=torch.int32).view(1, -1),
                            cache_seqlens=torch.tensor([5000], device="xpu", dtype=torch.int32), cu_seqlens_q=torch.tensor([0, 4096], device="xpu", dtype=torch.int32),
                            cu_seqlens_k_new=None, max_seqlen_q=4096, softmax_scale=0.0625, causal=True, k_descale=one, v_descale=one)
rep("sgl fp8 prefill attention")
from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
T, H, HV = 4096, 16, 48
qq = torch.randn(1, T, H, 128, device="xpu", dtype=torch.bfloat16); kk = torch.randn_like(qq)
vv = torch.randn(1, T, HV, 128, device="xpu", dtype=torch.bfloat16)
gg = -torch.rand(1, T, HV, device="xpu", dtype=torch.float32); bb = torch.rand(1, T, HV, device="xpu", dtype=torch.bfloat16)
try:
    chunk_gated_delta_rule(qq, kk, vv, g=gg, beta=bb, initial_state=torch.zeros(1, HV, 128, 128, device="xpu"), output_final_state=True,
                           cu_seqlens=torch.tensor([0, T], device="xpu", dtype=torch.int32), use_qk_l2norm_in_kernel=True)
    rep("triton GDN chunk prefill (first)")
except Exception as e:
    print("gdn chunk call failed:", repr(e)[:200])
