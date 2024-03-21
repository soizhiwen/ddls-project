# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import torch
from gluonts.torch.util import lagged_sequence_values

from uncond_ts_diff.arch import BackboneModel
from uncond_ts_diff.model.diffusion._base import TSDiffBase
from uncond_ts_diff.utils import get_lags_for_freq


class TSDiff(TSDiffBase):
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
        init_skip=True,
        lr=1e-3,
    ):
        super().__init__(
            backbone_parameters,
            timesteps=timesteps,
            diffusion_scheduler=diffusion_scheduler,
            context_length=context_length,
            prediction_length=prediction_length,
            num_feat_dynamic_real=num_feat_dynamic_real,
            num_feat_static_cat=num_feat_static_cat,
            num_feat_static_real=num_feat_static_real,
            cardinalities=cardinalities,
            freq=freq,
            normalization=normalization,
            use_features=use_features,
            use_lags=use_lags,
            lr=lr,
        )

        self.freq = freq
        if use_lags:
            # 获取一组适用的滞后时间点。这些滞后时间点表示在过去观察数据时应该考虑哪些具体的时间偏移量。
            self.lags_seq = get_lags_for_freq(freq)
            # 通过增加input_dim和output_dim来调整模型的主干网络（backbone）参数，确保网络能够处理包含滞后特征的输入数据。这是通过在原有的维度基础上加上滞后序列长度len(self.lags_seq)来实现的。(所以输入维度会变)
            backbone_parameters = backbone_parameters.copy()
            backbone_parameters["input_dim"] += len(self.lags_seq)
            backbone_parameters["output_dim"] += len(self.lags_seq)
        else:
            self.lags_seq = [0]
        self.input_dim = backbone_parameters["input_dim"]
        self.backbone = BackboneModel(
            **backbone_parameters,
            num_features=(
                self.num_feat_static_real
                + self.num_feat_static_cat
                + self.num_feat_dynamic_real
                + 1  # log_scale
            ),
            init_skip=init_skip,
        )
        # 指数移动平均（Exponential Moving Average，简称EMA），用于稳定和改进模型在训练过程中的表现和最终性能
        # 列表，用于存储EMA的更新率。每个元素代表一个特定的EMA更新率，用于控制旧模型权重与新模型权重合并时旧权重的保留比例。
        self.ema_rate = []  # [0.9999]
        self.ema_state_dicts = [
            # 用于创建模型权重的深拷贝，确保原始模型权重的更新不会影响到这些EMA权重
            copy.deepcopy(self.backbone.state_dict())
            # 为列表中的每个更新率创建一个权重拷贝
            for _ in range(len(self.ema_rate))
        ]

    def _extract_features(self, data):
        # 提取了除了最后self.context_length个时间点之外的所有历史数据。这部分数据被用作“先验”信息，可能用于模型的某些部分，如状态初始化、趋势估计等。
        prior = data["past_target"][:, : -self.context_length]
        # 提取了最后self.context_length个时间点的数据。这部分数据作为“上下文”信息，直接用于预测模型的当前步骤。
        context = data["past_target"][:, -self.context_length :]
        context_observed = data["past_observed_values"][
            :, -self.context_length :
        ]
        if self.normalization == "zscore":
            # self.scaler是一个函数或对象的方法，用于对context（上下文数据）进行标准化处理。它同时接收context_observed作为参数，context_observed指示了上下文数据中哪些点是实际观测到的。此外，data["stats"]包含了进行Z分数标准化所需的统计信息，如均值和标准差。Z分数标准化的公式通常为：Z = (x-u)/sigma
            scaled_context, scale = self.scaler(
                context, context_observed, data["stats"]
            )
        else:
            # 这可能涉及到最简单的缩放方法，如将数据缩放到[0, 1]范围内或其他类型的标准化。
            scaled_context, scale = self.scaler(context, context_observed)

        features = []

        # 将先前的观测数据（prior）除以之前获得的缩放因子（scale），以确保这部分数据与当前上下文数据（context）保持相同的标准化尺度。这是重要的，因为保持数据在同一尺度上有助于提高模型的学习效率和预测准确性。
        scaled_prior = prior / scale
        # 用相同的缩放因子（scale）来缩放未来的目标数据（future_target）。这确保了模型的输入（上下文和先前数据）和输出（未来目标）在相同的标准化尺度上，是训练模型时的常见做法。
        scaled_future = data["future_target"] / scale
        features.append(scale.log())

        x = torch.cat([scaled_context, scaled_future], dim=1)
        if data["feat_static_cat"] is not None:
            # Using from gluonts.torch.modules.feature import FeatureEmbedder in base
            features.append(self.embedder(data["feat_static_cat"]))
        if data["feat_static_real"] is not None:
            # 静态实数特征通常不需要嵌入操作，因为它们已经是以数值形式存在，可以直接被模型处理。
            features.append(data["feat_static_real"])
        static_feat = torch.cat(
            features,
            dim=1,
        )
        # 使用expand方法将静态特征沿着时间步长维度（即新添加的维度）扩展，以确保静态特征在每个时间步都有一个副本。x.shape[1]是时间序列的长度，-1表示保持其他维度大小不变。这样，对于每个时间步，模型都可以访问到相同的静态特征。
        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, x.shape[1], -1
        )

        features = [expanded_static_feat]

        time_features = []
        if data["past_time_feat"] is not None:
            time_features.append(
                data["past_time_feat"][:, -self.context_length :]
            )
        if data["future_time_feat"] is not None:
            time_features.append(data["future_time_feat"])
        features.append(torch.cat(time_features, dim=1))
        features = torch.cat(features, dim=-1)

        # 在时间序列分析中，滞后特征是从过去的数据中提取出来的，用于帮助模型学习数据的时间依赖性
        if self.use_lags:
            lags = lagged_sequence_values(
                self.lags_seq,
                scaled_prior,
                torch.cat([scaled_context, scaled_future], dim=1),
                dim=1,
            )
            x = torch.cat([x[:, :, None], lags], dim=-1)
        else:
            x = x[:, :, None]
        if not self.use_features:
            features = None

        return x, scale[:, :, None], features

    @torch.no_grad()
    def sample_n(
        self,
        num_samples: int = 1,
        return_lags: bool = False,
    ):
        # 在device设备上生成一个形状为(num_samples, seq_len, self.input_dim)的随机噪声张量samples，其中seq_len是上下文长度和预测长度的总和，self.input_dim是输入特征的维度。这些噪声样本将作为扩散过程的起始点。
        device = next(self.backbone.parameters()).device
        seq_len = self.context_length + self.prediction_length
        samples = torch.randn(
            (num_samples, seq_len, self.input_dim), device=device
        )

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            samples = self.p_sample(samples, t, i, features=None)

        samples = samples.cpu().numpy()

        if return_lags:
            return samples

        return samples[..., 0]

    # 定义了在每个训练批次结束后执行的操作，特别是在这里它用于更新模型的指数移动平均（EMA）权重。
    def on_train_batch_end(self, outputs, batch, batch_idx):
        for rate, state_dict in zip(self.ema_rate, self.ema_state_dicts):
            update_ema(state_dict, self.backbone.state_dict(), rate=rate)


def update_ema(target_state_dict, source_state_dict, rate=0.99):
    with torch.no_grad():
        for key, value in source_state_dict.items():
            ema_value = target_state_dict[key]
            ema_value.copy_(
                rate * ema_value + (1.0 - rate) * value.cpu(),
                # 参数允许异步复制，以提高效率。
                non_blocking=True,
            )
