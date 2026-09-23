#!/usr/bin/env bash
# Build exl3xpu/_C.so (SYCL/ESIMD torch op library). JIT (spir64) device code: the kernels are
# finalised by the Level Zero driver's IGC on first launch and cached.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
icpx -fsycl -fsycl-targets=spir64 -O3 -ffast-math -fPIC -std=c++17 -shared \
  -fsycl-device-code-split=per_kernel -D_GLIBCXX_USE_CXX11_ABI=1 ${EXL3_FLAGS:-} \
  -I csrc -I$T/include -I$T/include/torch/csrc/api/include \
  -x c++ csrc/exl3_ops.sycl -x none -o ${EXL3_OUT:-exl3xpu/_C.so} \
  -L$T/lib -Wl,-rpath,$T/lib -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu 2>&1 | grep -E "error" || true
ls -la ${EXL3_OUT:-exl3xpu/_C.so}
