import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

from torch.utils.cpp_extension import verify_ninja_availability
check = verify_ninja_availability()   # Does not return bool
print(check)

module_path = "/DiffusionMBIR/cudaop/op"
fused = load(
    "fused",
    sources=[
        os.path.join(module_path, "fused_bias_act.cpp"),
        os.path.join(module_path, "fused_bias_act_kernel.cu"),
    ],
)
