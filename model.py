from __future__ import annotations

from typing import Any

import torch

from essence_forge.core.models.simplified_fft_lwpt_se_tcn import SimplifiedFftLwptSeTCN
from essence_forge.features import build_fft_magnitude


class EssenceForgeTCN(SimplifiedFftLwptSeTCN):
    """Named export of the simplified FFT+LWPT+SE architecture for open release."""

    def _build_fft_magnitude(self, raw_input: torch.Tensor) -> torch.Tensor:
        return build_fft_magnitude(
            raw_input=raw_input,
            fft_mag_log1p=bool(self.fft_mag_log1p),
            fft_mag_norm=str(self.fft_mag_norm),
        )

    @classmethod
    def from_model_kwargs(cls, **kwargs: Any) -> "EssenceForgeTCN":
        return cls(**kwargs)
