import os
from pathlib import Path
import torch
from setuptools import setup
from torch.utils.cpp_extension import SyclExtension
import sys
sys.path.insert(0, os.environ.get("ESIMD_BUILD_HELPER", "/llm-scaler/vllm/custom-esimd-kernels-vllm"))
try:
    from esimd_build_extention import BuildExtension  # Intel llm-scaler's ESIMD-aware builder
except ImportError:
    from torch.utils.cpp_extension import BuildExtension

root = Path(__file__).parent.resolve()
torch_include = str(Path(torch.__file__).parent / "include")
sycl_flags = ["-O3", "-ffast-math", "-fsycl-device-code-split=per_kernel", f"-I{torch_include}"]
if os.environ.get("EXL3_AOT", "0") == "1":
    sycl_flags += ["-fsycl-targets=spir64_gen", "-Xs", "-device bmg"]
if os.environ.get("EXL3_ALL_CODEBOOKS"):
    sycl_flags += ["-DEXL3_ALL_CODEBOOKS"]

setup(
    name="exl3xpu",
    packages=["exl3xpu"],
    ext_modules=[SyclExtension(
        name="exl3xpu._C",
        sources=["csrc/exl3_ops.sycl"],
        include_dirs=[str(root / "csrc")],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"], "sycl": sycl_flags},
        extra_link_args=["-Wl,-rpath,$ORIGIN/../torch/lib"],
        py_limited_api=False,
    )],
    cmdclass={"build_ext": BuildExtension},
    entry_points={"vllm.general_plugins": ["exl3xpu = exl3xpu.vllm_plugin:register"]},
)
