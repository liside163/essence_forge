"""
timefreq.py

时频分支组件：
1. MultiResolutionSTFT: 多分辨率STFT谱图构建。
2. ResMultiScaleConv2D: 轻量多尺度2D残差卷积块。
3. TimeFreqEncoder2D: 多尺度谱图编码并压缩为1D时间序列特征。
4. TimeFreqAlign1D: 时频序列长度对齐到TCN时间轴。
5. LayerwiseFusionGate: 层级门控融合模块。
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_int_tuple(values: Iterable[int], name: str) -> Tuple[int, ...]:
    """
    将列表参数标准化为整数元组并做基础合法性校验。
    为什么这样做：
    1. 配置通常来自JSON list，统一为tuple便于不可变管理。
    2. 模块内部依赖固定长度的尺度列表，提前校验可避免运行时错误。
    """
    result = tuple(int(v) for v in values)
    if len(result) == 0:
        raise ValueError(f"{name} 不能为空")
    for item in result:
        if item <= 0:
            raise ValueError(f"{name} 中每个元素必须 > 0，当前={item}")
    return result


class MultiResolutionSTFT(nn.Module):
    """
    多分辨率STFT模块。

    输入:
    x: [B, C, T]

    输出:
    specs: List[[B, C, F_i, T_i]]

    说明：
    - 为了AMP稳定性，内部统一用float32执行FFT，再回到输入dtype。
    - center=False，保持与当前窗口化离线推理一致的时序语义。
    """

    def __init__(
        self,
        n_fft: Iterable[int],
        hop_lengths: Iterable[int],
        win_lengths: Iterable[int],
        log_scale: bool = True,
    ) -> None:
        super().__init__()
        self.n_fft = _to_int_tuple(n_fft, "n_fft")
        self.hop_lengths = _to_int_tuple(hop_lengths, "hop_lengths")
        self.win_lengths = _to_int_tuple(win_lengths, "win_lengths")
        if not (len(self.n_fft) == len(self.hop_lengths) == len(self.win_lengths)):
            raise ValueError("n_fft/hop_lengths/win_lengths 长度必须一致")
        for n_fft_i, win_i in zip(self.n_fft, self.win_lengths):
            if win_i > n_fft_i:
                raise ValueError(f"win_lengths 必须 <= n_fft，当前 win={win_i}, n_fft={n_fft_i}")
        self.log_scale = bool(log_scale)

        # 每个尺度各自注册窗函数，避免重复创建并确保设备迁移时自动同步。
        for idx, win_i in enumerate(self.win_lengths):
            window = torch.hann_window(win_i)
            self.register_buffer(f"window_{idx}", window, persistent=False)

    def _get_window(self, idx: int) -> torch.Tensor:
        """按尺度索引读取已注册窗函数。"""
        return getattr(self, f"window_{idx}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向计算多分辨率谱图。
        输入: [B, C, T]
        输出: List[[B, C, F_i, T_i]]
        """
        if x.ndim != 3:
            raise ValueError(f"MultiResolutionSTFT 输入必须是 [B, C, T]，实际={tuple(x.shape)}")
        batch_size, channels, seq_len = x.shape
        if seq_len < min(self.win_lengths):
            raise ValueError(
                f"序列长度过短：T={seq_len}，最小 win_length={min(self.win_lengths)}，无法做STFT"
            )

        x_fp32 = x.float().reshape(batch_size * channels, seq_len)
        specs: List[torch.Tensor] = []
        for idx, (n_fft_i, hop_i, win_i) in enumerate(zip(self.n_fft, self.hop_lengths, self.win_lengths)):
            spec = torch.stft(
                x_fp32,
                n_fft=n_fft_i,
                hop_length=hop_i,
                win_length=win_i,
                window=self._get_window(idx),
                center=False,
                return_complex=True,
            )
            mag = spec.abs()
            if self.log_scale:
                mag = torch.log1p(mag)
            mag = mag.view(batch_size, channels, mag.shape[-2], mag.shape[-1]).to(dtype=x.dtype)
            specs.append(mag)
        return specs


class ResMultiScaleConv2D(nn.Module):
    """
    轻量多尺度2D残差卷积块。

    输入:  [B, C_in, F, T_spec]
    输出:  [B, C_out, F, T_spec]

    设计意图：
    1. 用并行不同卷积核捕捉不同频谱纹理尺度。
    2. 通过残差连接提升训练稳定性。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Iterable[int] = (3, 5),
    ) -> None:
        super().__init__()
        kernel_tuple = _to_int_tuple(kernel_sizes, "kernel_sizes")
        branch_channels = max(out_channels // len(kernel_tuple), 8)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=branch_channels,
                        kernel_size=int(k),
                        padding=int(k) // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for k in kernel_tuple
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * len(kernel_tuple), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        输入: [B, C_in, F, T_spec]
        输出: [B, C_out, F, T_spec]
        """
        if x.ndim != 4:
            raise ValueError(f"ResMultiScaleConv2D 输入必须是 [B, C, F, T]，实际={tuple(x.shape)}")
        outputs = [branch(x) for branch in self.branches]
        fused = self.project(torch.cat(outputs, dim=1))
        out = self.out_act(fused + self.residual(x))
        return out


class TimeFreqEncoder2D(nn.Module):
    """
    多尺度时频编码器。

    输入:
    specs: List[[B, C_in, F_i, T_i]]

    输出:
    tf_seq: [B, C_out, T_pool]
    """

    def __init__(
        self,
        in_channels: int,
        num_scales: int,
        out_channels: int,
        scale_branch_channels: int,
        pool_freq_bins: int,
        pool_time_bins: int,
    ) -> None:
        super().__init__()
        if num_scales < 1:
            raise ValueError("num_scales 必须 >= 1")
        if pool_freq_bins <= 0 or pool_time_bins <= 0:
            raise ValueError("pool_freq_bins/pool_time_bins 必须 > 0")

        self.num_scales = int(num_scales)
        self.pool_freq_bins = int(pool_freq_bins)
        self.pool_time_bins = int(pool_time_bins)

        self.scale_encoders = nn.ModuleList(
            [
                ResMultiScaleConv2D(
                    in_channels=in_channels,
                    out_channels=scale_branch_channels,
                    kernel_sizes=(3, 5),
                )
                for _ in range(self.num_scales)
            ]
        )
        self.scale_pool = nn.AdaptiveAvgPool2d((self.pool_freq_bins, self.pool_time_bins))
        self.fuse_project = nn.Sequential(
            nn.Conv2d(scale_branch_channels * self.num_scales, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, specs: List[torch.Tensor]) -> torch.Tensor:
        """
        前向编码。
        输入: List[[B, C_in, F_i, T_i]]
        输出: [B, C_out, T_pool]
        """
        if len(specs) != self.num_scales:
            raise ValueError(f"输入尺度数与编码器不一致：期望={self.num_scales}, 实际={len(specs)}")

        encoded_per_scale: List[torch.Tensor] = []
        for encoder, spec in zip(self.scale_encoders, specs):
            # spec: [B, C_in, F_i, T_i]
            feat = encoder(spec)  # [B, C_scale, F_i, T_i]
            feat = self.scale_pool(feat)  # [B, C_scale, F_pool, T_pool]
            encoded_per_scale.append(feat)

        concat = torch.cat(encoded_per_scale, dim=1)  # [B, C_scale*num_scales, F_pool, T_pool]
        fused = self.fuse_project(concat)  # [B, C_out, F_pool, T_pool]

        # 沿频率维压缩为时序特征，供1D TCN融合。
        tf_seq = fused.mean(dim=2)  # [B, C_out, T_pool]
        return tf_seq


class TimeFreqAlign1D(nn.Module):
    """
    将时频分支输出对齐到TCN时间长度。
    输入:  [B, C, T_tf]
    输出:  [B, C, T_target]
    """

    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        线性插值对齐时间维。
        """
        if x.ndim != 3:
            raise ValueError(f"TimeFreqAlign1D 输入必须是 [B, C, T]，实际={tuple(x.shape)}")
        if target_length <= 0:
            raise ValueError(f"target_length 必须 > 0，当前={target_length}")
        if x.shape[-1] == target_length:
            return x
        return F.interpolate(x, size=target_length, mode="linear", align_corners=False)


class TimeFreqSpecAugmentor(nn.Module):
    """
    训练态频谱增强模块（SpecAugment + Spectral Tilt）。

    输入:
    specs: List[[B, C, F, T]]
    输出:
    与输入同形状的增强后谱图列表
    """

    def __init__(
        self,
        enable: bool = False,
        time_mask_enable: bool = True,
        time_mask_prob: float = 0.35,
        time_mask_width_range: Tuple[int, int] = (2, 6),
        freq_mask_enable: bool = True,
        freq_mask_prob: float = 0.35,
        freq_mask_width_range: Tuple[int, int] = (1, 2),
        spectral_tilt_enable: bool = True,
        spectral_tilt_prob: float = 0.2,
        spectral_tilt_db_range: float = 1.5,
    ) -> None:
        super().__init__()
        self.enable = bool(enable)
        self.time_mask_enable = bool(time_mask_enable)
        self.time_mask_prob = float(time_mask_prob)
        self.time_mask_width_range = (int(time_mask_width_range[0]), int(time_mask_width_range[1]))
        self.freq_mask_enable = bool(freq_mask_enable)
        self.freq_mask_prob = float(freq_mask_prob)
        self.freq_mask_width_range = (int(freq_mask_width_range[0]), int(freq_mask_width_range[1]))
        self.spectral_tilt_enable = bool(spectral_tilt_enable)
        self.spectral_tilt_prob = float(spectral_tilt_prob)
        self.spectral_tilt_db_range = float(spectral_tilt_db_range)

    @staticmethod
    def _normalize_width_range(width_range: Tuple[int, int]) -> Tuple[int, int]:
        low, high = int(width_range[0]), int(width_range[1])
        low = max(low, 1)
        high = max(high, low)
        return low, high

    def _apply_time_mask(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.time_mask_enable or self.time_mask_prob <= 0.0:
            return spec
        B, C, F, T = spec.shape
        if T <= 0:
            return spec
        low, high = self._normalize_width_range(self.time_mask_width_range)
        high = min(high, T)
        low = min(low, high)

        trigger = torch.rand((B, 1, 1, 1), device=spec.device) < self.time_mask_prob
        width = torch.randint(
            low=low,
            high=high + 1,
            size=(B, 1, 1, 1),
            device=spec.device,
        )
        start_max = max(T - low + 1, 1)
        start = torch.randint(
            low=0,
            high=start_max,
            size=(B, 1, 1, 1),
            device=spec.device,
        )
        t_index = torch.arange(T, device=spec.device).view(1, 1, 1, T)
        mask = (t_index >= start) & (t_index < (start + width))
        mask = mask & trigger
        return spec.masked_fill(mask.expand(-1, C, F, -1), 0.0)

    def _apply_freq_mask(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.freq_mask_enable or self.freq_mask_prob <= 0.0:
            return spec
        B, C, F, T = spec.shape
        if F <= 0:
            return spec
        low, high = self._normalize_width_range(self.freq_mask_width_range)
        high = min(high, F)
        low = min(low, high)

        trigger = torch.rand((B, 1, 1, 1), device=spec.device) < self.freq_mask_prob
        width = torch.randint(
            low=low,
            high=high + 1,
            size=(B, 1, 1, 1),
            device=spec.device,
        )
        start_max = max(F - low + 1, 1)
        start = torch.randint(
            low=0,
            high=start_max,
            size=(B, 1, 1, 1),
            device=spec.device,
        )
        f_index = torch.arange(F, device=spec.device).view(1, 1, F, 1)
        mask = (f_index >= start) & (f_index < (start + width))
        mask = mask & trigger
        return spec.masked_fill(mask.expand(-1, C, -1, T), 0.0)

    def _apply_spectral_tilt(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.spectral_tilt_enable or self.spectral_tilt_prob <= 0.0:
            return spec
        _, _, F, _ = spec.shape
        if F <= 1 or self.spectral_tilt_db_range <= 0.0:
            return spec

        B = spec.shape[0]
        trigger = (torch.rand((B, 1, 1, 1), device=spec.device) < self.spectral_tilt_prob).to(
            dtype=spec.dtype
        )
        tilt_db = (
            torch.rand((B, 1, 1, 1), device=spec.device, dtype=spec.dtype) * 2.0 - 1.0
        ) * self.spectral_tilt_db_range
        tilt_linear = torch.pow(
            torch.tensor(10.0, device=spec.device, dtype=spec.dtype),
            tilt_db / 20.0,
        )
        inv_tilt_linear = 1.0 / tilt_linear.clamp_min(1e-6)
        freq_axis = torch.linspace(0.0, 1.0, F, device=spec.device, dtype=spec.dtype).view(1, 1, F, 1)
        weights = inv_tilt_linear + freq_axis * (tilt_linear - inv_tilt_linear)
        weights = trigger * weights + (1.0 - trigger)
        return spec * weights

    def forward(self, specs: List[torch.Tensor]) -> List[torch.Tensor]:
        if not self.enable or not self.training:
            return specs
        out: List[torch.Tensor] = []
        for spec in specs:
            if spec.ndim != 4:
                out.append(spec)
                continue
            y = spec
            y = self._apply_time_mask(y)
            y = self._apply_freq_mask(y)
            y = self._apply_spectral_tilt(y)
            out.append(y)
        return out


class LayerwiseFusionGate(nn.Module):
    """
    层级门控融合模块。

    输入:
    main_feat: [B, C, T]
    tf_feat:   [B, C, T]

    输出:
    fused: [B, C, T]
    stats: 门控统计信息
    """

    def __init__(self, channels: int, alpha_init: float = 0.0, dropout: float = 0.1) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels 必须 > 0")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout 必须在 [0, 1) 范围内")
        self.channels = int(channels)
        self.gate_conv = nn.Conv1d(self.channels * 2, self.channels, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(self, main_feat: torch.Tensor, tf_feat: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        前向融合。
        输入: main_feat/tf_feat 都是 [B, C, T]
        输出: fused [B, C, T]
        """
        if main_feat.ndim != 3 or tf_feat.ndim != 3:
            raise ValueError(
                "LayerwiseFusionGate 输入必须都是 [B, C, T]"
                f"，实际 main={tuple(main_feat.shape)}, tf={tuple(tf_feat.shape)}"
            )
        if main_feat.shape != tf_feat.shape:
            raise ValueError(
                "LayerwiseFusionGate 需要主干与时频特征同形状，"
                f"实际 main={tuple(main_feat.shape)}, tf={tuple(tf_feat.shape)}"
            )
        gate = torch.sigmoid(self.gate_conv(torch.cat([main_feat, tf_feat], dim=1)))  # [B, C, T]
        injected = self.dropout(tf_feat)
        fused = main_feat + self.alpha.to(dtype=main_feat.dtype) * gate * injected
        stats = {
            "gate_mean": float(gate.mean().detach().cpu().item()),
            "gate_min": float(gate.min().detach().cpu().item()),
            "gate_max": float(gate.max().detach().cpu().item()),
            "alpha": float(self.alpha.detach().cpu().item()),
        }
        return fused, stats
