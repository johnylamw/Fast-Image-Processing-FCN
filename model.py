import torch.nn as nn
import torch

class CAN32(nn.Module):
    def __init__(self):
        super().__init__()

        # NOTE:
        # CAN32 network as described on pg. 13
        # L^0 and L^d = m x n x 3
        # m x n (image resolution) <- varies
        # Intermediate layer L^s = (1 <= s <= d - 1) = m x n x w

        dilation_scale = 2
        depth = 10
        width = 32
        dilation = 1

        # Layers #1
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, dilation=1, padding=1)
        self.norm1 = AdaptiveBatchNorm2D(num_features=width)

        # Layers #2 - #8 (dilation scales by 2)
        self.hidden_convs = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        for _ in range(depth - 3):
            dilation *= 2
            self.hidden_convs.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, dilation=dilation, padding=dilation))
            self.hidden_norms.append(AdaptiveBatchNorm2D(num_features=width))
        
        # Layer #9 (second to last) <- dilation=1
        self.layer_d_minus_1 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, dilation=1, padding=1)
        self.norm_d_minus_1 = AdaptiveBatchNorm2D(num_features=width)

        # Layer #10) output
        self.output = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=1, dilation=1)

        self.activation = nn.LeakyReLU(negative_slope=0.2) # alpha = 0.2 in paper

    def forward(self, x):
        # Layer 1
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.activation(x)

        # Layer 2  #8
        for conv, norm in zip(self.hidden_convs, self.hidden_norms):
            x = conv(x)
            x = norm(x)
            x = self.activation(x)

        # Layer 9
        x = self.layer_d_minus_1(x)
        x = self.norm_d_minus_1(x)
        x = self.activation(x)
    
        return self.output(x)

class AdaptiveBatchNorm2D(nn.Module):
    # lambda_s * x + mu_s * BN(X)
    # where lamba_s and mu_s are learnable parameters
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm2D, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.lambda_ = nn.Parameter(torch.ones(1))
        self.mu = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.lambda_ * x + self.mu * self.bn(x)