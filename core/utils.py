"""
utils.py

工具函数集合
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    设置全局随机种子，确保可复现性
    
    参数:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 为了完全可复现，可以启用以下设置（但会降低性能）
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def resolve_device(device_str: str) -> torch.device:
    """
    解析设备字符串并返回可用设备
    
    参数:
        device_str: "cuda" 或 "cpu"
        
    返回:
        torch.device 对象
    """
    if device_str.lower() == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("[WARN] CUDA 不可用，回退到 CPU")
            return torch.device("cpu")
    return torch.device("cpu")


def resolve_num_workers(num_workers: int, parallel_trials: int = 1) -> int:
    """
    解析并校验 DataLoader 的 num_workers。

    一些运行环境（容器/受限沙箱/WSL 配置）可能禁止创建 POSIX semaphore，
    这会导致 PyTorch DataLoader 在启动多进程 worker 时抛出 PermissionError。
    这里做一次轻量探测，不可用则回退到单进程加载（num_workers=0）。
    """

    n = int(num_workers)
    if n <= 0:
        return 0

    trial_count = max(1, int(parallel_trials))
    cpu_total = os.cpu_count() or 8
    n = min(n, max(2, cpu_total // trial_count))

    try:
        import multiprocessing as mp

        ctx = mp.get_context()
        lock = ctx.Lock()
        lock.acquire()
        lock.release()
        return n
    except PermissionError:
        print("[WARN] 当前环境不允许启用 DataLoader 多进程，已回退 num_workers=0")
        return 0
    except Exception as exc:
        print(f"[WARN] DataLoader 多进程初始化失败，已回退 num_workers=0: {exc}")
        return 0


def stable_seed_from_items(*items: Any, modulo: int = 2**32) -> int:
    """
    基于输入序列生成稳定随机种子，避免依赖 Python 内置 hash()。

    这样做的原因：
    1. 内置 hash() 默认带随机盐，不同进程结果可能不同。
    2. 训练/评估切窗若依赖该值，会破坏跨进程可复现性。

    参数:
        items: 任意可 repr 的输入项，按顺序参与哈希。
        modulo: 种子取模范围，默认 2**32。

    返回:
        稳定、非负的整数种子。
    """
    digest = hashlib.blake2b(digest_size=8)
    for item in items:
        digest.update(repr(item).encode("utf-8"))
        # 固定分隔符用于消除拼接歧义，保证哈希输入边界稳定。
        digest.update(b"\x1f")
    return int.from_bytes(digest.digest(), byteorder="big", signed=False) % int(modulo)


def build_dataloader_runtime_kwargs(
    *,
    num_workers: int,
    device: torch.device,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> Dict[str, Any]:
    """
    统一构建 DataLoader 运行时参数，确保训练与微调阶段行为一致。

    约束:
    - 仅在 CUDA 上开启 pin_memory/pin_memory_device；
    - 仅在 num_workers > 0 时设置 prefetch_factor 与 persistent_workers；
    - prefetch_factor 限制在 [2, 8]，避免极端值影响吞吐稳定性。
    """
    kwargs: Dict[str, Any] = {}
    use_cuda = device.type == "cuda"

    if bool(pin_memory and use_cuda):
        # `pin_memory_device` is deprecated in recent PyTorch versions.
        kwargs["pin_memory"] = True

    if int(num_workers) > 0:
        if bool(persistent_workers):
            kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = min(max(int(prefetch_factor), 2), 8)

    return kwargs


def read_json(path: Path) -> Any:
    """读取 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    """写入 JSON 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """将秒数格式化为可读时间字符串"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"
