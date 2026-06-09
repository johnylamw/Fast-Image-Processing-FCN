import torch.nn as nn
import torch
from torchvision.ops import DeformConv2d

class CAN(nn.Module):
    # norm_type = ["adaptive", "batch", "none"]
    def __init__(self, depth=10, width=32, norm_type="adaptive", use_deformconv=False):
        super().__init__()
        self.depth = depth
        self.width = width
        self.norm_type = norm_type
        self.use_deformconv = use_deformconv

        # NOTE:
        # CAN32 network as described on pg. 13
        # L^0 and L^d = m x n x 3
        # m x n (image resolution) <- varies
        # Intermediate layer L^s = (1 <= s <= d - 1) = m x n x w

        dilation = 1

        # Layers #1
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, dilation=1, padding=1)
        self.norm1 = self._make_norm(width, norm_type)

        # Layers #2 to #D - 2 (dilation scales by 2)
        self.hidden_convs = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        if use_deformconv:
            self.offset_convs = nn.ModuleList()
            for _ in range(depth - 3):
                self.offset_convs.append(nn.Conv2d(in_channels=width, out_channels=18, kernel_size=3, padding=1))
                self.hidden_convs.append(DeformConv2d(in_channels=width, out_channels=width, kernel_size=3, padding=1))
                self.hidden_norms.append(self._make_norm(width, norm_type))

            # Layer # D - 1 (second to last) <- dilation=1
            self.offset_d_minus_1 = nn.Conv2d(in_channels=width, out_channels=18, kernel_size=3, padding=1)
            self.layer_d_minus_1 = DeformConv2d(in_channels=width, out_channels=width, kernel_size=3, padding=1)
        else:
            for _ in range(depth - 3):
                dilation *= 2
                self.hidden_convs.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, dilation=dilation, padding=dilation))
                self.hidden_norms.append(self._make_norm(width, norm_type))
        
            # Layer # D - 1 (second to last) <- dilation=1
            self.layer_d_minus_1 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, dilation=1, padding=1)
            self.norm_d_minus_1 = self._make_norm(width, norm_type)

        # Layer #10) output
        self.output = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=1, dilation=1)

        self.activation = nn.LeakyReLU(negative_slope=0.2) # alpha = 0.2 in paper

    def _make_norm(self, num_features, norm_type):
        if norm_type == "adaptive":
            return AdaptiveBatchNorm2D(num_features=num_features)
        if norm_type == "batch":
            return nn.BatchNorm2d(num_features)
        if norm_type == "none":
            return nn.Identity()
        raise ValueError(f"Unknown norm_type: {norm_type}")

    def forward(self, x):
        # Layer 1
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.activation(x)


        # Layer 2 to D-2
        if self.use_deformconv:
            for offset_conv, conv, norm in zip(self.offset_convs, self.hidden_convs, self.hidden_norms):
                offset = offset_conv(x)
                x = conv(x, offset)
                x = norm(x)
                x = self.activation(x)
        else:
            for conv, norm in zip(self.hidden_convs, self.hidden_norms):
                x = conv(x)
                x = norm(x)
                x = self.activation(x)

        # Layer D-1
        if self.use_deformconv:
            offset = self.offset_d_minus_1(x)
            x = self.layer_d_minus_1(x, offset)
        else:
            x = self.layer_d_minus_1(x)
        x = self.norm_d_minus_1(x)
        x = self.activation(x)

        # Layer Output
        return self.output(x)

class CAN24AN(CAN):
    def __init__(self):
        super().__init__(depth=9, width=24, norm_type="adaptive", use_deformconv=False)

class CAN32AN(CAN):
    def __init__(self):
        super().__init__(depth=10, width=32, norm_type="adaptive", use_deformconv=False)

class CAN24AND(CAN):
    def __init__(self):
        super().__init__(depth=9, width=24, norm_type="adaptive", use_deformconv=True)

class CAN32AND(CAN):
    def __init__(self):
        super().__init__(depth=10, width=32, norm_type="adaptive", use_deformconv=True)

class CAN32(CAN):
    def __init__(self):
        super().__init__(depth=10, width=32, norm_type="none", use_deformconv=False)

class CAN32BN(CAN):
    def __init__(self):
        super().__init__(depth=10, width=32, norm_type="batch", use_deformconv=False)

class AdaptiveBatchNorm2D(nn.Module):
    # lambda_s * x + mu_s * BN(X)
    # where lamba_s and mu_s are learnable parameters
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super(AdaptiveBatchNorm2D, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.lambda_ = nn.Parameter(torch.ones(1))
        self.mu = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.lambda_ * x + self.mu * self.bn(x)
