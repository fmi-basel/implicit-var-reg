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

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import wandb
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.byol import byol_loss_func, byol_loss_func_l2
from solo.losses.isoloss import isoloss_func, isoloss_func_l2
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params


class DirectPred(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements DirectPred ().

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of projected features.
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                dp_alpha (float): power of the eigenvalues for the prediction matrix.
                dp_normalize (bool): whether to normalize the eigenvalues.
                iso (bool): whether to use isotropic loss instead of negative cosine similarity loss.
                dp_tau (float): momentum for the correlation matrix.
                eps_iso (float): epsilon for adding to the eigenvalues.
                log_eigvals (bool): whether to log the eigenvalues.
                use_l2 (bool): whether to use l2 loss instead of negative cosine similarity loss.
        """

        super().__init__(cfg)

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        self.dp_alpha: float = cfg.method_kwargs.dp_alpha
        self.dp_normalize: bool = cfg.method_kwargs.dp_normalize
        self.iso: bool = cfg.method_kwargs.iso
        self.dp_tau: float = cfg.method_kwargs.dp_tau
        self.eps_iso: float = cfg.method_kwargs.eps_iso
        self.log_eigvals: bool = cfg.method_kwargs.log_eigvals
        self.use_l2: bool = cfg.method_kwargs.use_l2

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # initialize correlation matrix of all zeros
        self.register_buffer("C", torch.zeros(proj_output_dim, proj_output_dim))

        # intialize eigenvalues as zeros
        self.register_buffer("s", torch.zeros(proj_output_dim))

        self.eigval_table = None

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(DirectPred, DirectPred).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.dp_alpha")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.dp_normalize")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.iso")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.dp_tau")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.log_eigvals")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.use_l2")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])

        # precondition the Zs
        zn = z.detach() / np.sqrt(z.shape[0])  # move division by batchsize here to avoid overflow

        # update correlation matrix
        # corr_matrix = torch.matmul(z.T, z) / z.shape[0] # (bs, dim) * (bs, dim) -> (dim, dim)
        corr_matrix = torch.matmul(zn.T, zn)  # (bs, dim) * (bs, dim) -> (dim, dim)
        self.C = self.C * self.dp_tau + corr_matrix * (1 - self.dp_tau)

        if self.dp_alpha == 2.0 :
            self.s = torch.linalg.eigvals(self.C).real
            if self.dp_normalize:
                # divide by max eigenvalue
                normalized_C = self.C / self.s.max() + self.eps_iso * torch.eye(self.C.shape[0])

            # calculate Wp
            Wp = torch.matmul(normalized_C, normalized_C)

            if self.iso:
                Wpsqrt = normalized_C
                psqrt = torch.matmul(z, Wpsqrt)
                out.update({"psqrt": psqrt})

        else:
            # calculate eigenvalues and eigenvectors and take real part
            s, U = torch.linalg.eig(self.C)
            self.s = s.real
            U = U.real
            s = self.s

            if self.dp_normalize:
                # normalize eigenvalues
                s = torch.clamp(s, min=0) / s.max()
                s = s + self.eps_iso
            else:
                # catch nan values, clip below to zero and above to 1e6
                s = torch.nan_to_num(s, nan=0.0)
                s = torch.clamp(s, min=0, max=1e6)

            # raise eigenvalues to the power of alpha
            s = s**self.dp_alpha if self.dp_alpha != 1.0 else s

            # # calculate Wp
            Wp = torch.matmul(U, torch.matmul(torch.diag(s), U.T))

            if self.iso:
                Wpsqrt = torch.matmul(U, torch.matmul(torch.diag(s**0.5), U.T))
                psqrt = torch.matmul(z, Wpsqrt)
                out.update({"psqrt": psqrt})

        # calculate prediction
        p = torch.matmul(z, Wp)

        out.update({"z": z, "p": p})

        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])

        # update correlation matrix
        corr_matrix = (
            torch.matmul(z.T, z).detach() / z.shape[0]
        )  # (bs, dim) * (bs, dim) -> (dim, dim)
        self.C = self.C * self.dp_tau + corr_matrix * (1 - self.dp_tau)

        # calculate eigenvalues and eigenvectors and take real part
        s, U = torch.linalg.eigh(self.C)
        s = s.real
        U = U.real

        if self.dp_normalize:
            # normalize eigenvalues
            s = torch.clamp(s, min=0) / s.max() + self.eps_iso
        else:
            # catch nan values, clip below to zero and above to 1e6
            s = torch.nan_to_num(s, nan=0.0)
            s = torch.clamp(s, min=0, max=1e6)

        # raise eigenvalues to the power of alpha
        s = s**self.dp_alpha

        # rotate projections into the eigenbasis
        z = torch.matmul(z, U)

        # calculate prediction in eigenspace by multiplying z with the eigenvalues
        p = z * s

        out.update({"z": z, "p": p})
        return out

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        """Performs the forward pass of the momentum backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of
                the parent and the momentum projected features.
        """

        out = super().momentum_forward(X)
        z = self.momentum_projector(out["feats"])

        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        Z = out["z"]
        P = out["p"]
        Z_momentum = out["momentum_z"]

        # ------- negative cosine similarity loss -------
        neg_cos_sim = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                if self.iso:
                    if self.use_l2:
                        neg_cos_sim += isoloss_func_l2(
                            P[v2], out["psqrt"][v2], Z[v2], Z_momentum[v1]
                        )
                    else:
                        neg_cos_sim += isoloss_func(P[v2], out["psqrt"][v2], Z[v2], Z_momentum[v1])
                else:
                    if self.use_l2:
                        neg_cos_sim += byol_loss_func_l2(P[v2], Z_momentum[v1])
                    else:
                        neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])

        # calculate std of features and participation ratio
        with torch.no_grad():
            z_std = F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()
            part_ratio = (self.s.sum() ** 2) / (self.s**2).sum()

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
            "eigval_metrics/eigval_min": self.s.min(),
            "eigval_metrics/eigval_max": self.s.max(),
            "eigval_metrics/eigval_mean": self.s.mean(),
            "eigval_metrics/eigval_std": self.s.std(),
            "eigval_metrics/part_ratio": part_ratio,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss

    def on_train_epoch_end(self) -> None:
        if self.log_eigvals:
            # create table if first epoch
            if self.current_epoch == 0:
                eigval_list = self.s.reshape(1, -1).tolist()
                self.eigval_table = wandb.Table(
                    data=eigval_list,
                    columns=["eigval_{}".format(i) for i in range(1, len(self.s) + 1)],
                )
            else:
                eigval_list = self.s.tolist()
                self.eigval_table.add_data(*eigval_list)

        return super().on_train_epoch_end()

    def on_train_end(self) -> None:
        if self.log_eigvals:
            wandb.log({"eigvals": self.eigval_table})
        return super().on_train_end()
