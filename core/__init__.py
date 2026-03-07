"""
uav_tl 包

UAV TCN迁移学习核心模块
"""

__all__ = [
    "build_index",
    "save_splits",
    "MissionLoader",
    "SourceDomainDataset",
    "TargetDomainDataset",
    "DeterministicWindowDataset",
    "collate_padded",
]


def __getattr__(name: str):
    if name in {"build_index", "save_splits"}:
        from essence_forge.core.rflymad_index import build_index, save_splits

        return {"build_index": build_index, "save_splits": save_splits}[name]

    if name == "MissionLoader":
        from essence_forge.core.rflymad_io import MissionLoader

        return MissionLoader

    if name in {
        "SourceDomainDataset",
        "TargetDomainDataset",
        "DeterministicWindowDataset",
        "collate_padded",
    }:
        from essence_forge.core.datasets import (
            SourceDomainDataset,
            TargetDomainDataset,
            DeterministicWindowDataset,
            collate_padded,
        )

        return {
            "SourceDomainDataset": SourceDomainDataset,
            "TargetDomainDataset": TargetDomainDataset,
            "DeterministicWindowDataset": DeterministicWindowDataset,
            "collate_padded": collate_padded,
        }[name]

    raise AttributeError(f"module 'uav_tl' has no attribute {name!r}")
