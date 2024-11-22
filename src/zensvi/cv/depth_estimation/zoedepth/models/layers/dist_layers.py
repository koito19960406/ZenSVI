# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
import torch.nn as nn


def log_binom(n, k, eps=1e-7):
    """Calculate the logarithm of the binomial coefficient using Stirling's approximation.

    Args:
        n (torch.Tensor): The total number of items.
        k (torch.Tensor): The number of items to choose.
        eps (float, optional): A small value to prevent log(0). Defaults to 1e-7.

    Returns:
        torch.Tensor: The logarithm of the binomial coefficient.
    """
    n = n + eps
    k = k + eps
    return n * torch.log(n) - k * torch.log(k) - (n - k) * torch.log(n - k + eps)


class LogBinomial(nn.Module):
    """Log Binomial distribution model.

    This class computes the log binomial distribution for a given number of classes.

    Args:
        n_classes (int, optional): Number of output classes. Defaults to 256.
        act (callable, optional): Activation function to apply. Defaults to torch.softmax.
    """

    def __init__(self, n_classes=256, act=torch.softmax):
        super().__init__()
        self.K = n_classes
        self.act = act
        self.register_buffer("k_idx", torch.arange(0, n_classes).view(1, -1, 1, 1))
        self.register_buffer("K_minus_1", torch.Tensor([self.K - 1]).view(1, -1, 1, 1))

    def forward(self, x, t=1.0, eps=1e-4):
        """Compute the log binomial distribution for the input probabilities.

        Args:
            x (torch.Tensor): Input probabilities of shape (N, C, H, W).
            t (float, optional): Temperature of the distribution. Defaults to 1.0.
            eps (float, optional): Small number for numerical stability. Defaults to 1e-4.

        Returns:
            torch.Tensor: Log binomial distribution of shape (N, C, H, W).
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # make it nchw

        one_minus_x = torch.clamp(1 - x, eps, 1)
        x = torch.clamp(x, eps, 1)
        y = (
            log_binom(self.K_minus_1, self.k_idx)
            + self.k_idx * torch.log(x)
            + (self.K - 1 - self.k_idx) * torch.log(one_minus_x)
        )
        return self.act(y / t, dim=1)


class ConditionalLogBinomial(nn.Module):
    """Conditional Log Binomial distribution model.

    This class implements a conditional log binomial distribution that takes both a main feature
    and a condition feature as input. It outputs a probability distribution over n_classes bins.

    Args:
        in_features (int): Number of input channels in the main feature.
        condition_dim (int): Number of input channels in the condition feature.
        n_classes (int, optional): Number of output classes/bins. Defaults to 256.
        bottleneck_factor (int, optional): Factor to reduce hidden dimension size. Defaults to 2.
        p_eps (float, optional): Small epsilon value for numerical stability. Defaults to 1e-4.
        max_temp (float, optional): Maximum temperature for output distribution. Defaults to 50.
        min_temp (float, optional): Minimum temperature for output distribution. Defaults to 1e-7.
        act (callable, optional): Activation function to apply. Defaults to torch.softmax.

    Attributes:
        p_eps (float): Small epsilon value for numerical stability.
        max_temp (float): Maximum temperature for output distribution.
        min_temp (float): Minimum temperature for output distribution.
        log_binomial_transform (LogBinomial): Transform to compute log binomial distribution.
        mlp (nn.Sequential): Multi-layer perceptron to process concatenated features.
    """

    def __init__(
        self,
        in_features,
        condition_dim,
        n_classes=256,
        bottleneck_factor=2,
        p_eps=1e-4,
        max_temp=50,
        min_temp=1e-7,
        act=torch.softmax,
    ):
        super().__init__()
        self.p_eps = p_eps
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.log_binomial_transform = LogBinomial(n_classes, act=act)
        bottleneck = (in_features + condition_dim) // bottleneck_factor
        self.mlp = nn.Sequential(
            nn.Conv2d(
                in_features + condition_dim,
                bottleneck,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            # 2 for p linear norm, 2 for t linear norm
            nn.Conv2d(bottleneck, 2 + 2, kernel_size=1, stride=1, padding=0),
            nn.Softplus(),
        )

    def forward(self, x, cond):
        """Forward pass through the Conditional Log Binomial model.

        Args:
            x (torch.Tensor): Main feature of shape (N, C, H, W).
            cond (torch.Tensor): Condition feature of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output log binomial distribution.
        """
        pt = self.mlp(torch.concat((x, cond), dim=1))
        p, t = pt[:, :2, ...], pt[:, 2:, ...]

        p = p + self.p_eps
        p = p[:, 0, ...] / (p[:, 0, ...] + p[:, 1, ...])

        t = t + self.p_eps
        t = t[:, 0, ...] / (t[:, 0, ...] + t[:, 1, ...])
        t = t.unsqueeze(1)
        t = (self.max_temp - self.min_temp) * t + self.min_temp

        return self.log_binomial_transform(p, t)
