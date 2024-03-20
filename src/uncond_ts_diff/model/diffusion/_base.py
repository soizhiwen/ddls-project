# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.modules.scaler import MeanScaler, NOPScaler

from uncond_ts_diff.utils import extract

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "future_time_feat",
]


class TSDiffBase(pl.LightningModule):
    def __init__(
        self,
        backbone_parameters,
        timesteps,
        diffusion_scheduler,
        context_length,
        prediction_length,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinalities=None,
        freq=None,
        normalization="none",
        use_features=False,
        use_lags=True,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.timesteps = timesteps
        self.betas = diffusion_scheduler(timesteps)
        self.sqrt_one_minus_beta = torch.sqrt(1.0 - self.betas)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.logs = {}
        self.normalization = normalization
        if normalization == "mean":
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)
        if cardinalities is None:
            cardinalities = [1]
        self.embedder = FeatureEmbedder(
            # 包含每个静态类别特征的类别数（或称基数）。基数是指该特征中不同类别的数量。例如，如果有一个特征是“星期几”，那么它的基数就是7，因为一周有7天。
            cardinalities=cardinalities,
            # 为每个类别特征指定嵌入维度, 限制在最大50维
            embedding_dims=[min(50, (cat + 1) // 2) for cat in cardinalities],
        )
        self.time_features = (
            # 这个函数返回的是一个时间特征函数的列表。每个函数都是一个特征提取器，能够接收一个时间点并返回一个与该特征相关的数值。
            time_features_from_frequency_str(freq) if freq is not None else []
        )

        self.num_feat_dynamic_real = (
            # 基本特征 + 用户定义的特征 + 时间特征
            # 每个从时间频率字符串生成的时间特征都被认为是一个动态实数特征
            1 + num_feat_dynamic_real + len(self.time_features)
        )
        self.num_feat_static_cat = max(num_feat_static_cat, 1)
        self.num_feat_static_real = max(num_feat_static_real, 1)

        self.use_features = use_features
        self.use_lags = use_lags

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.losses_running_mean = torch.ones(timesteps, requires_grad=False)
        self.lr = lr
        self.best_crps = np.inf

    def _extract_features(self, data):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # 会在一定条件下降低学习率
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=int(1e12)
        )
        return [optimizer], {"scheduler": scheduler, "monitor": "train_loss"}

    def log(self, name, value, **kwargs):
        super().log(name, value, **kwargs)
        if isinstance(value, torch.Tensor):
            # 将其转换为一个Python标量
            value = value.detach().cpu().item()
        if name not in self.logs:
            self.logs[name] = [value]
        else:
            self.logs[name].append(value)

    def get_logs(self):
        logs = self.logs
        logs["epochs"] = list(range(self.current_epoch))
        return pd.DataFrame.from_dict(logs)

    def q_sample(self, x_start, t, noise=None):
        device = next(self.backbone.parameters()).device
        if noise is None:
            # 生成标准正态分布（均值为0，方差为1）的噪声
            noise = torch.randn_like(x_start, device=device)
        sqrt_alphas_cumprod_t = extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def p_losses(
        self,
        x_start,
        t,
        # 额外的特征，可以用于损失计算。
        features=None,
        # 如果提供，这个噪声会被用来直接参与损失计算，而不是从数据中重新生成。
        noise=None,
        loss_type="l2",
        reduction="mean",
    ):
        device = next(self.backbone.parameters()).device
        if noise is None:
            noise = torch.randn_like(x_start, device=device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.backbone(x_noisy, t, features)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise, reduction=reduction)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise, reduction=reduction)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(
                noise, predicted_noise, reduction=reduction
            )
        else:
            raise NotImplementedError()

        return loss, x_noisy, predicted_noise

    @torch.no_grad()
    def p_sample(self, x, t, t_index, features=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        predicted_noise = self.backbone(x, t, features)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        
    # DDIM，Deterministic Denoising Diffusion Implicit Models
    @torch.no_grad()
    def p_sample_ddim(self, x, t, features=None, noise=None):
        if noise is None:
            noise = self.backbone(x, t, features)
        sqrt_alphas_cumprod_prev_t = extract(
            self.alphas_cumprod_prev, t, x.shape
        ).sqrt()
        sqrt_one_minus_alphas_cumprod_prev_t = extract(
            1 - self.alphas_cumprod_prev, t, x.shape
        ).sqrt()
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        x0pointer = (
            # 首先从当前的噪声数据x中减去根据当前时间步估计的噪声成分（sqrt_one_minus_alphas_cumprod_t * noise），这是为了去除噪声的影响并尽可能恢复接近于原始数据的状态。
            # 然后，通过与sqrt_alphas_cumprod_prev_t / sqrt_alphas_cumprod_t相乘来调整尺度，这个比率考虑了在连续时间步中α的累积乘积（即整个扩散过程中稳定性因子的累积影响），以确保恢复的数据尺度正确。
            sqrt_alphas_cumprod_prev_t
            * (x - sqrt_one_minus_alphas_cumprod_t * noise)
            / sqrt_alphas_cumprod_t
        )
        # xtpointer代表在给定时间步t，由噪声直接贡献的部分。考虑了从开始到前一时间步累积的α因子对噪声影响的调整。
        xtpointer = sqrt_one_minus_alphas_cumprod_prev_t * noise
        # 生成了在时间步t处，对原始数据x0的一个整体估计。
        return x0pointer + xtpointer

    # 这种方法允许在纯DDIM（Deterministic Denoising Diffusion Implicit Models，eta=0）和纯DDPM（Denoising Diffusion Probabilistic Models，eta=1）之间进行插值
    @torch.no_grad()
    def p_sample_genddim(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
        # 上一个时间步的标识（如果有的话）。
        t_prev: Optional[torch.Tensor] = None,
        # 插值参数，允许在DDPM和DDIM之间进行插值。eta=0时，模型行为类似于DDIM，即更加确定性；eta=1时，模型行为类似于DDPM，引入了更多的随机性。
        eta: float = 0.0,
        # 可选的额外特征，可以提供给模型以辅助恢复原始数据。
        features=None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generalized DDIM step that interpolates between
        DDPM (eta=1) and DDIM (eta=0).

        Args:
            x (torch.Tensor): _description_
            t (torch.Tensor): _description_
            features (_type_, optional): _description_. Defaults to None.
            noise (Optional[torch.Tensor], optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        if noise is None:
            noise = self.backbone(x, t, features)
        if t_prev is None:
            t_prev = t - 1

        alphas_cumprod_t = extract(self.alphas_cumprod, t, x.shape)
        alphas_cumprod_prev_t = (
            extract(self.alphas_cumprod, t_prev, x.shape)
            if t_index > 0
            else torch.ones_like(alphas_cumprod_t)
        )
        sqrt_alphas_cumprod_prev_t = alphas_cumprod_prev_t.sqrt()

        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)

        x0pointer = (
            sqrt_alphas_cumprod_prev_t
            * (x - sqrt_one_minus_alphas_cumprod_t * noise)
            / sqrt_alphas_cumprod_t
        )
        c1 = (
            eta
            * (
                (1 - alphas_cumprod_t / alphas_cumprod_prev_t)
                * (1 - alphas_cumprod_prev_t)
                / (1 - alphas_cumprod_t)
            ).sqrt()
        )
        c2 = ((1 - alphas_cumprod_prev_t) - c1**2).sqrt()
        return x0pointer + c1 * torch.randn_like(x) + c2 * noise

    @torch.no_grad()
    def sample(self, noise, features=None):
        device = next(self.backbone.parameters()).device
        batch_size, length, ch = noise.shape
        seq = noise
        seqs = [seq.cpu()]

        for i in reversed(range(0, self.timesteps)):
            seq = self.p_sample(
                seq,
                torch.full((batch_size,), i, device=device, dtype=torch.long),
                i,
                features,
            )
            # 每次逆向步骤后，将当前步骤生成的序列添加到seqs列表中
            seqs.append(seq.cpu().numpy())

        return np.stack(seqs, axis=0)

    def fast_denoise(self, xt, t, features=None, noise=None):
        if noise is None:
            noise = self.backbone(xt, t, features)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, xt.shape
        )
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, xt.shape)
        return (
            xt - sqrt_one_minus_alphas_cumprod_t * noise
        ) / sqrt_alphas_cumprod_t

    def forward(self, x, mask):
        raise NotImplementedError()

    def training_step(self, data, idx):
        assert self.training is True
        device = next(self.backbone.parameters()).device
        if isinstance(data, dict):
            # 如果data是一个字典，那么使用self._extract_features(data)来提取特征
            x, _, features = self._extract_features(data)
        else:
            # 如果不是，假设数据直接是输入张量x，并可能通过self.scaler进行缩放处理。
            x, _ = self.scaler(data, torch.ones_like(data))

        # 随机生成一个与批次大小相同的时间步向量t，这些时间步用于模拟扩散过程中的不同阶段。每个数据点会被分配一个随机的时间步。
        t = torch.randint(
            0, self.timesteps, (x.shape[0],), device=device
        ).long()
        elbo_loss, xt, noise = self.p_losses(x, t, features, loss_type="l2")
        return {
            "loss": elbo_loss,
            "elbo_loss": elbo_loss,
        }

    def training_epoch_end(self, outputs):
        epoch_loss = sum(x["loss"] for x in outputs) / len(outputs)
        elbo_loss = sum(x["elbo_loss"] for x in outputs) / len(outputs)
        self.log("train_loss", epoch_loss)
        self.log("train_elbo_loss", elbo_loss)

    def validation_step(self, data, idx):
        device = next(self.backbone.parameters()).device
        if isinstance(data, dict):
            x, _, features = self._extract_features(data)
        else:
            x, features = data, None
        t = torch.randint(
            0, self.timesteps, (x.shape[0],), device=device
        ).long()
        elbo_loss, xt, noise = self.p_losses(x, t, features, loss_type="l2")
        return {
            "loss": elbo_loss,
            "elbo_loss": elbo_loss,
        }

    def validation_epoch_end(self, outputs):
        epoch_loss = sum(x["loss"] for x in outputs) / len(outputs)
        elbo_loss = sum(x["elbo_loss"] for x in outputs) / len(outputs)
        self.log("valid_loss", epoch_loss)
        self.log("valid_elbo_loss", elbo_loss)
