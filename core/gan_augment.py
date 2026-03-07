"""
gan_augment.py

面向“易错分类别”的定向 GAN 数据增强。

设计目标：
1. 从历史评估混淆矩阵自动挖掘 hard classes（低 recall 且支持度足够）。
2. 对 hard classes 训练改进 WGAN-GP：
   - Wasserstein + Gradient Penalty（稳定训练）
   - 频谱一致性约束（保持故障频域特征）
   - 时序平滑约束（抑制高频抖动伪影）
3. 推理阶段不直接替换真实样本，而是与真实窗口做凸组合，保证物理可解释性。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - 仅在缺依赖环境触发
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ClassDifficulty:
    """
    单类难度统计（来自混淆矩阵）。
    """

    class_id: int
    support: int
    correct: int
    recall: float
    misclassified: int


@dataclass(frozen=True)
class GanTrainingConfig:
    """
    WGAN-GP 训练配置（按单类独立训练）。
    """

    latent_dim: int = 96
    hidden_dim: int = 256
    train_steps: int = 180
    critic_steps: int = 3
    batch_size: int = 64
    lr_generator: float = 2e-4
    lr_discriminator: float = 4e-4
    gp_lambda: float = 10.0
    spectral_lambda: float = 1.0
    smooth_lambda: float = 0.02
    synth_bank_size: int = 512
    min_windows_per_class: int = 24
    seed: int = 2026
    device: str = "cpu"


def build_synthetic_bank_cache_key(
    hard_classes: Sequence[int],
    class_windows: Mapping[int, np.ndarray],
    config: GanTrainingConfig,
) -> str:
    """
    构建 targeted GAN synthetic bank 的稳定缓存键。
    """
    window_shapes = {
        str(int(class_id)): tuple(int(x) for x in np.asarray(windows).shape)
        for class_id, windows in class_windows.items()
    }
    signature = {
        "hard_classes": [int(x) for x in sorted(set(hard_classes))],
        "window_shapes": window_shapes,
        "config": {
            "latent_dim": int(config.latent_dim),
            "hidden_dim": int(config.hidden_dim),
            "train_steps": int(config.train_steps),
            "critic_steps": int(config.critic_steps),
            "batch_size": int(config.batch_size),
            "lr_generator": float(config.lr_generator),
            "lr_discriminator": float(config.lr_discriminator),
            "gp_lambda": float(config.gp_lambda),
            "spectral_lambda": float(config.spectral_lambda),
            "smooth_lambda": float(config.smooth_lambda),
            "synth_bank_size": int(config.synth_bank_size),
            "min_windows_per_class": int(config.min_windows_per_class),
            "seed": int(config.seed),
            "device": str(config.device),
        },
    }
    raw = json.dumps(signature, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def save_synthetic_bank_cache(
    cache_path: str | Path,
    synthetic_bank_by_class: Mapping[int, np.ndarray],
    train_stats: Mapping[int, Mapping[str, float]],
    metadata: Optional[Mapping[str, object]] = None,
) -> None:
    """
    保存 synthetic bank 缓存到 npz。
    """
    path = Path(cache_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized_bank: Dict[int, np.ndarray] = {}
    for class_id, bank in synthetic_bank_by_class.items():
        arr = np.asarray(bank, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] > 0:
            normalized_bank[int(class_id)] = arr

    class_ids = sorted(normalized_bank.keys())
    payload: Dict[str, np.ndarray] = {
        "class_ids": np.asarray(class_ids, dtype=np.int64),
        "train_stats_json": np.asarray(
            json.dumps({str(int(k)): dict(v) for k, v in train_stats.items()}, ensure_ascii=True),
            dtype=np.str_,
        ),
        "metadata_json": np.asarray(
            json.dumps(dict(metadata or {}), ensure_ascii=True),
            dtype=np.str_,
        ),
    }
    for class_id in class_ids:
        payload[f"class_{class_id}"] = normalized_bank[class_id]

    np.savez_compressed(path, **payload)


def load_synthetic_bank_cache(
    cache_path: str | Path,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, float]], Dict[str, object]]:
    """
    从 npz 缓存读取 synthetic bank。
    """
    path = Path(cache_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"synthetic bank 缓存不存在: {path}")

    bank: Dict[int, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as payload:
        class_ids = payload["class_ids"].astype(np.int64).tolist()
        for class_id in class_ids:
            key = f"class_{int(class_id)}"
            if key not in payload:
                continue
            arr = np.asarray(payload[key], dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] > 0:
                bank[int(class_id)] = arr

        train_stats_raw = json.loads(str(payload["train_stats_json"].item()))
        metadata_raw = json.loads(str(payload["metadata_json"].item()))

    train_stats: Dict[int, Dict[str, float]] = {}
    for class_id_str, stats in train_stats_raw.items():
        class_id = int(class_id_str)
        train_stats[class_id] = {str(k): float(v) for k, v in dict(stats).items()}
    metadata = dict(metadata_raw)
    return bank, train_stats, metadata


def load_or_train_synthetic_bank(
    cache_path: str | Path,
    class_windows: Mapping[int, np.ndarray],
    config: GanTrainingConfig,
    trainer: Optional[
        Callable[[Mapping[int, np.ndarray], GanTrainingConfig], Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, float]]]]
    ] = None,
    force_retrain: bool = False,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, float]], bool]:
    """
    优先从缓存加载 synthetic bank，未命中时训练并回写缓存。

    返回：
    - synthetic_bank
    - train_stats
    - cache_hit
    """
    path = Path(cache_path).expanduser().resolve()
    if path.exists() and not force_retrain:
        try:
            bank, stats, _ = load_synthetic_bank_cache(path)
            if len(bank) > 0:
                return bank, stats, True
        except Exception as exc:
            print(f"[GAN增强] 缓存读取失败，将回源训练: {exc}")

    train_fn = trainer if trainer is not None else train_targeted_wgan_synthetic_bank
    bank, stats = train_fn(class_windows, config)
    if len(bank) > 0:
        metadata = {
            "cache_version": 1,
            "seed": int(config.seed),
            "device": str(config.device),
        }
        save_synthetic_bank_cache(
            cache_path=path,
            synthetic_bank_by_class=bank,
            train_stats=stats,
            metadata=metadata,
        )
    return bank, stats, False


if nn is not None:
    class _Generator(nn.Module):
        def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            return self.net(z)


    class _Discriminator(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x).view(-1)
else:
    class _Generator:  # pragma: no cover - 仅在缺依赖环境触发
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("train_targeted_wgan_synthetic_bank 需要安装 torch")


    class _Discriminator:  # pragma: no cover - 仅在缺依赖环境触发
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("train_targeted_wgan_synthetic_bank 需要安装 torch")


def _validate_confusion_matrix(confusion_matrix: np.ndarray) -> np.ndarray:
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion_matrix 必须为方阵，当前 shape={cm.shape}")
    if np.any(cm < 0):
        raise ValueError("confusion_matrix 不能包含负值")
    return cm


def rank_classes_by_difficulty(
    confusion_matrix: np.ndarray,
    exclude_classes: Sequence[int] = (),
) -> Tuple[ClassDifficulty, ...]:
    """
    按分类难度排序（recall 升序、误分量降序、支持度降序）。
    """

    cm = _validate_confusion_matrix(confusion_matrix)
    excluded = {int(x) for x in exclude_classes}
    rows = []
    for class_id in range(cm.shape[0]):
        if class_id in excluded:
            continue
        support = int(cm[class_id].sum())
        correct = int(cm[class_id, class_id])
        misclassified = int(max(support - correct, 0))
        recall = float(correct / support) if support > 0 else 0.0
        rows.append(
            ClassDifficulty(
                class_id=int(class_id),
                support=support,
                correct=correct,
                recall=recall,
                misclassified=misclassified,
            )
        )

    rows.sort(key=lambda x: (x.recall, -x.misclassified, -x.support, x.class_id))
    return tuple(rows)


def select_hard_classes_from_confusion(
    confusion_matrix: np.ndarray,
    top_k: int,
    min_support: int,
    exclude_classes: Sequence[int] = (),
) -> list[int]:
    """
    基于混淆矩阵选取 hard classes。
    """

    if top_k <= 0:
        return []
    if min_support < 0:
        raise ValueError("min_support 必须 >= 0")

    ranked = rank_classes_by_difficulty(
        confusion_matrix=confusion_matrix,
        exclude_classes=exclude_classes,
    )
    selected = [row.class_id for row in ranked if row.support >= int(min_support)]
    return selected[: int(top_k)]


def select_hard_classes_from_eval_json(
    eval_json_path: str | Path,
    top_k: int,
    min_support: int,
    exclude_classes: Sequence[int] = (),
) -> list[int]:
    """
    从评估结果 JSON（含 confusion_matrix）自动选择 hard classes。
    """

    p = Path(eval_json_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"评估文件不存在: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if "confusion_matrix" not in payload:
        raise KeyError(f"{p} 中缺少字段 confusion_matrix")
    return select_hard_classes_from_confusion(
        confusion_matrix=np.asarray(payload["confusion_matrix"], dtype=np.float64),
        top_k=top_k,
        min_support=min_support,
        exclude_classes=exclude_classes,
    )


class TargetedSyntheticBankAugmentor:
    """
    基于“类专属 synthetic bank”的定向增强器。

    说明：
    - `real_window` 与 `synthetic_window` 做凸组合：
      `x_aug = alpha * real + (1-alpha) * synthetic`
    - `alpha` 可控，保证增强样本仍锚定真实物理轨迹。
    """

    def __init__(
        self,
        synthetic_bank_by_class: Mapping[int, np.ndarray],
        apply_prob: float,
        blend_ratio_min: float,
        blend_ratio_max: float,
    ) -> None:
        self.synthetic_bank_by_class: Dict[int, np.ndarray] = {}
        for class_id, bank in synthetic_bank_by_class.items():
            arr = np.asarray(bank, dtype=np.float32)
            if arr.ndim != 3 or arr.shape[0] == 0:
                continue
            self.synthetic_bank_by_class[int(class_id)] = arr

        self.apply_prob = float(np.clip(apply_prob, 0.0, 1.0))
        lo = float(np.clip(blend_ratio_min, 0.0, 1.0))
        hi = float(np.clip(blend_ratio_max, 0.0, 1.0))
        self.blend_ratio_min = min(lo, hi)
        self.blend_ratio_max = max(lo, hi)

    def enabled_classes(self) -> Tuple[int, ...]:
        return tuple(sorted(self.synthetic_bank_by_class.keys()))

    def maybe_augment(
        self,
        class_id: int,
        real_window: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        对指定类别样本执行概率增强，其他类别原样返回。
        """

        real = np.asarray(real_window, dtype=np.float32)
        bank = self.synthetic_bank_by_class.get(int(class_id))
        if bank is None or bank.shape[0] == 0:
            return real
        if rng.random() > self.apply_prob:
            return real

        idx = int(rng.integers(0, bank.shape[0]))
        synth = bank[idx]
        if synth.shape != real.shape:
            return real

        alpha = float(rng.uniform(self.blend_ratio_min, self.blend_ratio_max))
        out = alpha * real + (1.0 - alpha) * synth
        return out.astype(np.float32, copy=False)


def _gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand((real.size(0), 1), device=real.device, dtype=real.dtype)
    interp = alpha * real + (1.0 - alpha) * fake
    interp.requires_grad_(True)
    score = discriminator(interp)
    grads = torch.autograd.grad(
        outputs=score,
        inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_norm = grads.norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


def _spectral_profile(x: torch.Tensor) -> torch.Tensor:
    # x: [B, T, C]
    amp = torch.fft.rfft(x, dim=1).abs().mean(dim=2)  # [B, F]
    return torch.log1p(amp).mean(dim=0)  # [F]


def _spectral_consistency_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(_spectral_profile(fake) - _spectral_profile(real)))


def _temporal_smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] <= 1:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    diff = x[:, 1:, :] - x[:, :-1, :]
    return torch.mean(diff.pow(2))


def _normalize_feature_stats(
    fake: np.ndarray,
    real: np.ndarray,
) -> np.ndarray:
    """
    将生成样本做逐特征重标定，逼近真实类样本一阶/二阶统计。
    """

    eps = 1e-6
    fake_mean = fake.mean(axis=0, keepdims=True)
    fake_std = fake.std(axis=0, keepdims=True)
    real_mean = real.mean(axis=0, keepdims=True)
    real_std = real.std(axis=0, keepdims=True)
    fake_std = np.maximum(fake_std, eps)
    real_std = np.maximum(real_std, eps)
    normalized = (fake - fake_mean) / fake_std
    return normalized * real_std + real_mean


def train_targeted_wgan_synthetic_bank(
    class_windows: Mapping[int, np.ndarray],
    config: GanTrainingConfig,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, float]]]:
    """
    对每个类别独立训练改进 WGAN-GP，生成 synthetic bank。

    输入：
    - `class_windows[class_id]`: [N, T, C]（建议已完成标准化）

    返回：
    - `synthetic_bank_by_class`: 每类生成窗口 [M, T, C]
    - `train_stats`: 每类训练统计（用于日志）
    """

    if torch is None or nn is None:
        raise ModuleNotFoundError("train_targeted_wgan_synthetic_bank 需要安装 torch")

    requested = str(config.device).strip().lower()
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] targeted_gan.device=cuda 但 CUDA 不可用，回退到 CPU")
        requested = "cpu"
    device = torch.device(requested)
    rng = np.random.default_rng(int(config.seed))
    torch.manual_seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))

    synth_bank: Dict[int, np.ndarray] = {}
    train_stats: Dict[int, Dict[str, float]] = {}

    for class_id, windows in class_windows.items():
        arr = np.asarray(windows, dtype=np.float32)
        if arr.ndim != 3:
            continue
        n, t, c = arr.shape
        if n < int(config.min_windows_per_class):
            continue

        x_real = torch.from_numpy(arr.reshape(n, t * c)).to(device)
        input_dim = int(t * c)
        generator = _Generator(
            latent_dim=int(config.latent_dim),
            hidden_dim=int(config.hidden_dim),
            output_dim=input_dim,
        ).to(device)
        discriminator = _Discriminator(
            input_dim=input_dim,
            hidden_dim=int(config.hidden_dim),
        ).to(device)

        opt_g = torch.optim.Adam(
            generator.parameters(),
            lr=float(config.lr_generator),
            betas=(0.5, 0.9),
        )
        opt_d = torch.optim.Adam(
            discriminator.parameters(),
            lr=float(config.lr_discriminator),
            betas=(0.5, 0.9),
        )

        batch_size = min(int(config.batch_size), n)
        last_d_loss = 0.0
        last_g_loss = 0.0

        for _ in range(int(config.train_steps)):
            for _ in range(int(config.critic_steps)):
                idx = torch.from_numpy(rng.integers(0, n, size=batch_size, endpoint=False)).to(device)
                real_batch = x_real[idx]
                z = torch.randn((batch_size, int(config.latent_dim)), device=device)
                fake_batch = generator(z).detach()

                d_real = discriminator(real_batch).mean()
                d_fake = discriminator(fake_batch).mean()
                gp = _gradient_penalty(discriminator, real_batch, fake_batch)
                d_loss = d_fake - d_real + float(config.gp_lambda) * gp

                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
                last_d_loss = float(d_loss.detach().cpu().item())

            idx = torch.from_numpy(rng.integers(0, n, size=batch_size, endpoint=False)).to(device)
            real_batch = x_real[idx]
            z = torch.randn((batch_size, int(config.latent_dim)), device=device)
            fake_batch = generator(z)
            fake_seq = fake_batch.view(batch_size, t, c)
            real_seq = real_batch.view(batch_size, t, c)

            adv_loss = -discriminator(fake_batch).mean()
            spec_loss = _spectral_consistency_loss(fake_seq, real_seq)
            smooth_loss = _temporal_smoothness_loss(fake_seq)
            g_loss = (
                adv_loss
                + float(config.spectral_lambda) * spec_loss
                + float(config.smooth_lambda) * smooth_loss
            )

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
            last_g_loss = float(g_loss.detach().cpu().item())

        gen_count = int(max(1, config.synth_bank_size))
        fake_chunks = []
        generator.eval()
        with torch.no_grad():
            generated = 0
            while generated < gen_count:
                cur = min(256, gen_count - generated)
                z = torch.randn((cur, int(config.latent_dim)), device=device)
                fake = generator(z).view(cur, t, c).detach().cpu().numpy().astype(np.float32)
                fake_chunks.append(fake)
                generated += cur
        fake_all = np.concatenate(fake_chunks, axis=0)
        fake_all = _normalize_feature_stats(fake=fake_all, real=arr)

        synth_bank[int(class_id)] = fake_all.astype(np.float32, copy=False)
        train_stats[int(class_id)] = {
            "train_windows": float(n),
            "synthetic_windows": float(gen_count),
            "last_d_loss": float(last_d_loss),
            "last_g_loss": float(last_g_loss),
        }

    return synth_bank, train_stats


def build_class_difficulty_from_eval_json(
    eval_json_path: str | Path,
    exclude_classes: Sequence[int] = (),
) -> Tuple[ClassDifficulty, ...]:
    """
    从评估 JSON 构建排序后的类难度统计。
    """

    p = Path(eval_json_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"评估文件不存在: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if "confusion_matrix" not in payload:
        raise KeyError(f"{p} 中缺少字段 confusion_matrix")
    return rank_classes_by_difficulty(
        confusion_matrix=np.asarray(payload["confusion_matrix"], dtype=np.float64),
        exclude_classes=exclude_classes,
    )
