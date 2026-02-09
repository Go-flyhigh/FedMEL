import torch
from torch import nn
from typing import Optional, Tuple
import math
class ConvBlock(nn.Module):
    """ConvBlock that expands channels, concatenates, and reduces.

    Training-time structure:
      - 3x3 conv + BN (global residual carrier, in -> in)
      - local branch A: 3x3 conv + BN -> 1x1 conv + BN (in -> C_expand -> C_expand)
      - local branch B: 1x1 conv + BN -> 1x1 conv + BN (in -> C_expand -> C_expand)
      concat -> 1x1 reduce + BN (2*C_expand -> in)

    The module supports fused upload via get_equivalent_kernel_bias
    """

    def __init__(
        self,
        in_channels: int,
        expand_channels: int = 16,
        kernel_size: int = 5,
        padding: Optional[int] = None,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.expand_channels = expand_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size // 2) if padding is None else padding
        self.stride = stride

        self.branch_3x3_global = self._conv_bn(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
        )
        # Local branch A: 3x3 -> 1x1
        self.branch_3x3_local_pre = self._conv_bn(
            in_channels,
            expand_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
        )
        self.branch_3x3_local_post = self._conv_bn(
            expand_channels,
            expand_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # Local branch B: 1x1 -> 1x1
        self.branch_1x1_local_pre = self._conv_bn(
            in_channels,
            expand_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
        )
        self.branch_1x1_local_post = self._conv_bn(
            expand_channels,
            expand_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # Reduce after concatenation
        self.reduce_1x1 = self._conv_bn(
            2 * expand_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv_bn = nn.BatchNorm2d(in_channels)

    def _conv_bn(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> nn.Sequential:
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
        ]
        layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_global = self.branch_3x3_global(x)

        # Local branch A: 3x3 -> 1x1
        y_local1 = self.branch_3x3_local_pre(x)
        y_local1 = self.branch_3x3_local_post(y_local1)

        # Local branch B: 1x1 -> 1x1
        y_local2 = self.branch_1x1_local_pre(x)
        y_local2 = self.branch_1x1_local_post(y_local2)

        y_local_cat = torch.cat([y_local1, y_local2], dim=1)
        y_red = self.reduce_1x1(y_local_cat)
        out = self.conv_bn(y_red + y_global)
        return out

    def get_equivalent_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_global, bias_global = self._fuse_conv_bn(self.branch_3x3_global)
        # ---- Global branch ----
        reduce_weight, reduce_bias = self._fuse_conv_bn(self.reduce_1x1)
        # ---- Local branch A ----
        k1_3, b1_3 = self._fuse_conv_bn(self.branch_3x3_local_pre)
        k2_3, b2_3 = self._fuse_conv_bn(self.branch_3x3_local_post)
        kernel_local1, bias_local1 = self._fuse_sequential_1x1(k1_3, b1_3, k2_3, b2_3)

        # ---- Local branch B ----
        k1_1, b1_1 = self._fuse_conv_bn(self.branch_1x1_local_pre)
        k2_1, b2_1 = self._fuse_conv_bn(self.branch_1x1_local_post)
        kernel_local2, bias_local2 = self._fuse_sequential_1x1(k1_1, b1_1, k2_1, b2_1)

        # Combine each local branch with its slice of the reduce 1x1 (cat -> reduce)
        kernel_local1, bias_local1 = self._combine_reduce_branch(
            reduce_weight,
            kernel_local1,
            bias_local1,
            0,
            self.expand_channels,
        )
        kernel_local2, bias_local2 = self._combine_reduce_branch(
            reduce_weight,
            kernel_local2,
            bias_local2,
            self.expand_channels,
            2 * self.expand_channels,
        )

        kernel = kernel_local1 + kernel_local2 + kernel_global
        bias = bias_local1 + bias_local2 + reduce_bias + bias_global

        kernel, bias = self._fuse_bn_to_kernel_bias(kernel, bias, self.conv_bn)

        return kernel, bias

    def _fuse_conv_bn(self, branch: nn.Sequential) -> tuple[torch.Tensor, torch.Tensor]:
        conv = branch[0]
        bn = branch[1] if len(branch) > 1 else None

        if bn is None:
            weight = conv.weight
            bias = conv.bias if conv.bias is not None else torch.zeros_like(weight[:, 0, 0, 0])
            return weight, bias

        std = torch.sqrt(bn.running_var + bn.eps)
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        fused_weight = conv.weight * t
        fused_bias = bn.bias - bn.running_mean * bn.weight / std
        if conv.bias is not None:
            fused_bias += conv.bias * bn.weight / std
        return fused_weight, fused_bias

    def _fuse_bn_to_kernel_bias(
        self,
        kernel: torch.Tensor,  
        bias: torch.Tensor,    
        bn: nn.BatchNorm2d,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse a BN layer applied AFTER the conv output:
            y = BN(conv(x; kernel, bias))
        Returns an equivalent conv (kernel_fused, bias_fused).
        """
        std = torch.sqrt(bn.running_var + bn.eps)         
        t = (bn.weight / std)                              
        c = bn.bias - bn.running_mean * bn.weight / std    

        kernel_fused = kernel * t.reshape(-1, 1, 1, 1)
        bias_fused = bias * t + c
        return kernel_fused, bias_fused

    def _fuse_sequential_1x1(
        self,
        kernel1: torch.Tensor,
        bias1: torch.Tensor,
        kernel2: torch.Tensor,
        bias2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse two conv(+BN) blocks in sequence where the second conv is 1x1.
        Parameters
            kernel1, bias1:
                Equivalent (W, b) of the first block with shape (M, I, k, k) and (M,).
            kernel2, bias2:
                Equivalent (W, b) of the second block with shape (O, M, 1, 1) and (O,).
        Returns
            fused_kernel, fused_bias:
                Equivalent (W, b) for the composition with shape (O, I, k, k) and (O,).
        """
        k2 = kernel2[:, :, 0, 0]  # (O, M)
        fused_kernel = torch.einsum("om, mihw -> oihw", k2, kernel1)
        fused_bias = bias2 + torch.einsum("om, m -> o", k2, bias1)
        return fused_kernel, fused_bias


    def _combine_reduce_branch(
        self,
        reduce_weight: torch.Tensor,
        branch_kernel: torch.Tensor,
        branch_bias: torch.Tensor,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_slice = reduce_weight[:, start:end, 0, 0]
        #calculate the 3channels kernel and bias (both kernel and bias are all effected by reduce1*1 weight)
        kernel = torch.einsum("oc, cihw -> oihw", weight_slice, branch_kernel)
        bias = torch.einsum("oc, c -> o", weight_slice, branch_bias)
        if kernel.size(2) != self.kernel_size or kernel.size(3) != self.kernel_size:
            pad_total = self.kernel_size // 2
            kernel = nn.functional.pad(kernel, [pad_total, pad_total, pad_total, pad_total])
        return kernel, bias

    def set_global_branch_from_equivalent(
        self, residual_kernel: torch.Tensor, residual_bias: torch.Tensor
    ) -> None:
        """Write residual directly into the global 3x3 branch."""

        main_conv = self.branch_3x3_global[0]
        main_conv.weight.data.copy_(residual_kernel)
        if main_conv.bias is not None:
            main_conv.bias.data.copy_(residual_bias)

        if len(self.branch_3x3_global) > 1:
            bn = self.branch_3x3_global[1]
            _set_bn_to_identity(bn)
            bn.bias.data.copy_(residual_bias)
        else:
            main_conv.bias = torch.nn.Parameter(residual_bias.clone())

def _set_bn_to_identity(bn: nn.BatchNorm2d) -> None:
    bn.weight.data.fill_(math.sqrt(1.0 + bn.eps))
    bn.bias.data.zero_()
    bn.running_mean.zero_()
    bn.running_var.fill_(1.0)
    if hasattr(bn, "num_batches_tracked"):
        bn.num_batches_tracked.zero_()

class LinearBlock(nn.Module):
    """LinearBlock with two branches 500-d."""
    def __init__(self,in_features: int = 500) -> None:
        super().__init__()
        self.in_channels = in_features
        self.branch_global = nn.Linear(in_features, in_features, bias=True)
        self.branch_local = nn.Linear(in_features, in_features, bias=True)
        self.LinearBlock_ln = nn.LayerNorm(in_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_global = self.branch_global(x)
        y_local = self.branch_local(x)
        out = self.LinearBlock_ln(y_global + y_local)
        return out

    def get_equivalent_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight_eq = self.branch_global.weight + self.branch_local.weight
        bias_eq = self.branch_global.bias + self.branch_local.bias
        return weight_eq, bias_eq

    def set_global_branch_from_equivalent(self, target_weight: torch.Tensor, target_bias: torch.Tensor) -> None:
        self.branch_global.weight.data.copy_(target_weight)
        self.branch_global.bias.data.copy_(target_bias)
        
def build_fused_state_dict(model: nn.Module,) -> dict:
    fused_sd: dict[str, torch.Tensor] = {}
    fusable_prefixes = tuple(
        name + "."
        for name, module in model.named_modules()
        if callable(getattr(module, "get_equivalent_weight_bias", None))
    )
    for key, param in model.state_dict().items():
        if ".LinearBlock_ln." in key:
            fused_sd[key] = param.detach().clone()
            continue

        if fusable_prefixes and key.startswith(fusable_prefixes):
            continue
        fused_sd[key] = param.detach().clone()

    for name, module in model.named_modules():
        if callable(getattr(module, "get_equivalent_weight_bias", None)):
            weight, bias = module.get_equivalent_weight_bias()
            fused_sd[f"{name}.weight"] = weight.detach().clone()
            fused_sd[f"{name}.bias"] = bias.detach().clone()
    return fused_sd

def load_fused_weights_into_heteros(
    model: nn.Module,  
    fused_sd: dict,
) -> None:
    fusable_prefixes = tuple(
        name + "."
        for name, module in model.named_modules()
        if callable(getattr(module, "get_equivalent_weight_bias", None))
    )
    model_sd = model.state_dict()
    for name, param in model_sd.items(): 
        if name not in fused_sd : 
            continue
        if ".LinearBlock_ln." in name:
            param.copy_(fused_sd[name])
            continue
        if fusable_prefixes and name.startswith(fusable_prefixes): 
            continue
        param.copy_(fused_sd[name])
    for name, module in model.named_modules():
        if callable(getattr(module, "get_equivalent_weight_bias", None)):
            weight_key, bias_key = f"{name}.weight", f"{name}.bias"
            if hasattr(module, "set_global_branch_from_equivalent"):
                module.set_global_branch_from_equivalent(fused_sd[weight_key], fused_sd[bias_key])
