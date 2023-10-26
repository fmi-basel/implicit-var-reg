# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn.functional as F


def isoloss_func(
    p: torch.Tensor, psqrt: torch.Tensor, z: torch.Tensor, zt: torch.Tensor
) -> torch.Tensor:
    """Computes the isotropic loss given batch of predicted features p and projected momentum features z.

    Args:
        p (torch.Tensor): NxD Tensor containing predicted features from view 1
        psqrt (torch.Tensor): NxD Tensor containing predicted features from view 1 using sqrt prediction
        z (torch.Tensor): NxD Tensor containing projected momentum features from view 1
        zt (torch.Tensor): NxD Tensor containing projected momentum features from view 2
    Returns:
        torch.Tensor: isotropic loss.
    """

    sim = (z * zt.detach()).sum(dim=1) / (
        torch.norm(p, dim=1) * torch.norm(zt, dim=1)
    ).detach() - 0.5 * (
        (p * zt).sum(dim=1) / (torch.norm(p, dim=1) ** 3 * torch.norm(zt, dim=1))
    ).detach() * torch.norm(
        psqrt, dim=1
    ) ** 2

    return -2 * sim.mean()


def isoloss_func_l2(
    p: torch.Tensor, psqrt: torch.Tensor, z: torch.Tensor, zt: torch.Tensor
) -> torch.Tensor:
    """Computes the isotropic euclidean loss form given batch of predicted features p and projected momentum features z.

    Args:
        p (torch.Tensor): NxD Tensor containing predicted features from view 1
        psqrt (torch.Tensor): NxD Tensor containing predicted features from view 1 using sqrt prediction
        z (torch.Tensor): NxD Tensor containing projected momentum features from view 1
        zt (torch.Tensor): NxD Tensor containing projected momentum features from view 2
    Returns:
        torch.Tensor: isotropic loss.
    """

    return torch.mean(torch.sum((z - (zt + z - p).detach()) ** 2, dim=1))
