#!/usr/bin/env bash
# Compile + AOT-link csrc/exl3_ops.sycl for BMG with extra -D flags, to bisect IGC failures.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
cd /w
icpx -fsycl ${TARGETS:--fsycl-targets=spir64_gen -Xs "-device bmg"} -O3 -ffast-math -fPIC -std=c++17 -shared \
  ${SPLIT--fsycl-device-code-split=per_kernel} \
  -I/w/csrc -I$T/include -I$T/include/torch/csrc/api/include -I/usr/include/python3.12 \
  "$@" -x c++ csrc/exl3_ops.sycl -x none -o /tmp/probe.so -L$T/lib -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu > /tmp/probe.log 2>&1
rc=$?
grep -E " error|Build|rror:|kernel" /tmp/probe.log | grep -v warning | head -8
echo "exit=$rc"
