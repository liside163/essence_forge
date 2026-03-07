from __future__ import annotations

import torch

from essence_forge.core.models.masking import masked_max_pooling, masked_mean_pooling


def build_fft_magnitude(
    raw_input: torch.Tensor,
    fft_mag_log1p: bool = True,
    fft_mag_norm: str = "per_sample_channel",
) -> torch.Tensor:
    normalized = str(fft_mag_norm).strip().lower()
    if normalized not in {"none", "per_sample_channel", "global"}:
        raise ValueError("fft_mag_norm must be one of none/per_sample_channel/global")

    fft_in = raw_input.float()
    fft_mag = torch.abs(torch.fft.fft(fft_in, dim=-1))
    if fft_mag_log1p:
        fft_mag = torch.log1p(fft_mag)
    if normalized == "per_sample_channel":
        denom = fft_mag.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        fft_mag = fft_mag / denom
    elif normalized == "global":
        denom = fft_mag.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        fft_mag = fft_mag / denom
    return fft_mag.to(dtype=raw_input.dtype)


def pool_branch_tokens(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    use_mixed_pooling: bool,
) -> torch.Tensor:
    pooled_mean = masked_mean_pooling(tokens, lengths=lengths)
    if not use_mixed_pooling:
        return pooled_mean
    pooled_max = masked_max_pooling(tokens, lengths=lengths)
    return torch.cat([pooled_mean, pooled_max], dim=1)
