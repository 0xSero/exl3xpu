import torch
g = 2 ** 30
def rep(tag):
    torch.xpu.synchronize(); f, t = torch.xpu.mem_get_info(); r = torch.xpu.memory_reserved()
    print(f"{tag:40s} free {f/g:6.2f}  torch reserved {r/g:6.2f}  non-torch {(t-f-r)/g:6.2f} GiB", flush=True)
rep("start")
from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
T, H, HV, S = 4096, 16, 48, 16
pool = torch.zeros(S, HV, 128, 128, device="xpu", dtype=torch.float16)
rep("state pool")
for T in (4096, 1000, 4096):
    qq = torch.randn(1, T, H, 128, device="xpu", dtype=torch.bfloat16); kk = torch.randn_like(qq)
    vv = torch.randn(1, T, HV, 128, device="xpu", dtype=torch.bfloat16)
    gg = -torch.rand(1, T, HV, device="xpu", dtype=torch.float32); bb = torch.rand(1, T, HV, device="xpu", dtype=torch.bfloat16)
    chunk_gated_delta_rule(qq, kk, vv, g=gg, beta=bb, initial_state=pool, initial_state_indices=torch.tensor([3], device="xpu", dtype=torch.int32),
                           cu_seqlens=torch.tensor([0, T], device="xpu", dtype=torch.int32), head_first=False, use_qk_l2norm_in_kernel=True)
    rep(f"triton GDN chunk prefill T={T}")
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update as fr
for B in (1, 8):
    q = torch.randn(1, B * 4, H, 128, device="xpu", dtype=torch.bfloat16); k = torch.randn_like(q); v = torch.randn(1, B * 4, HV, 128, device="xpu", dtype=torch.bfloat16)
    a = torch.randn(B * 4, HV, device="xpu", dtype=torch.bfloat16); b = torch.randn_like(a)
    fr(A_log=torch.zeros(HV, device="xpu"), a=a, dt_bias=torch.zeros(HV, device="xpu"), softplus_beta=1.0, softplus_threshold=20.0,
       q=q, k=k, v=v, b=b, initial_state_source=pool, initial_state_indices=torch.arange(B, device="xpu", dtype=torch.int32),
       cu_seqlens=torch.arange(B + 1, device="xpu", dtype=torch.int32) * 4, use_qk_l2norm_in_kernel=True, disable_state_update=True)
    rep(f"triton GDN recurrent verify B={B}")
