"""
train.py

婧愬煙棰勮缁冮€昏緫

鑱岃矗:
1. 鍦ㄦ簮鍩熸暟鎹笂璁粌 TCN 妯″瀷
2. Early Stopping锛堟寜 val macro-F1锛?3. 淇濆瓨鏈€浣?checkpoint
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

import sys

from essence_forge.core.runtime_config import CFG
from essence_forge.core.datasets import (
    SourceDomainDataset,
    DeterministicWindowDataset,
    collate_padded,
)
from essence_forge.core.precomputed_dataset import PrecomputedDataset
from essence_forge.core.rflymad_io import MissionLoader
from essence_forge.core.losses import (
    ConfusionAwareLoss,
    SupConLoss,
    CenterLoss,
    AdaptiveCostSensitiveLoss,
    FocalLoss,
    LabelSmoothingCrossEntropy,
    LDAMLoss,
)
from essence_forge.core.cost_sensitive import (
    build_cost_tensor_from_best_costs,
    extract_supported_hparams,
    load_optuna_cost_result,
)
from essence_forge.core.model_checkpoint import (
    build_model_init_kwargs_from_cfg,
    normalize_model_class_name,
    resolve_model_class,
)
from essence_forge.core.utils import (
    build_dataloader_runtime_kwargs,
    read_json,
    resolve_device,
    resolve_num_workers,
    set_global_seed,
)


@dataclass
class TrainResult:
    """璁粌缁撴灉璁板綍"""
    model_name: str
    best_epoch: int
    best_val_macro_f1: float
    checkpoint_path: str
    history: List[Dict]


def _load_source_artifacts(run_dir: Path):
    """
    鍔犺浇婧愬煙璁粌鎵€闇€鐨勬暟鎹?    
    杩斿洖:
        (train_records, val_records, mean, std)
    """
    train_path = run_dir / "split_source_train.json"
    val_path = run_dir / "split_source_val.json"
    stats_path = run_dir / "source_stats.json"
    
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Missing split_source_*.json. Please run index first.")
    if not stats_path.exists():
        raise FileNotFoundError("Missing source_stats.json. Please run index first.")
    
    train_records = read_json(train_path)
    val_records = read_json(val_path)
    stats = read_json(stats_path)
    
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    
    return train_records, val_records, mean, std


def _compute_class_weights(train_records: List[Dict]) -> torch.Tensor:
    """
    璁＄畻绫诲埆鏉冮噸锛堢敤浜?class-balanced loss锛?    
    鏀寔涓ょ妯″紡:
    - inverse_freq: w = 1 / freq
    - inverse_log: w = 1 / log(offset + freq)
    """
    counts = np.zeros((CFG.num_classes,), dtype=np.float64)
    for rec in train_records:
        counts[int(rec["class_id"])] += 1
    counts = np.maximum(counts, 1.0)  # 闃叉闄ら浂
    
    if CFG.class_weight_mode == "inverse_freq":
        weights = 1.0 / counts
    elif CFG.class_weight_mode == "inverse_log":
        weights = 1.0 / np.log(CFG.inverse_log_offset + counts)
    else:
        weights = np.ones_like(counts)
    
    # 褰掍竴鍖栧埌骞冲潎涓?1
    weights = weights / weights.mean()
    return torch.tensor(weights.astype(np.float32))


def _build_weighted_sampler_from_labels(
    labels: List[int],
    num_samples: Optional[int] = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """
    鍩轰簬鏍囩搴忓垪鏋勫缓 WeightedRandomSampler銆?
    璁捐璇存槑锛?    - 涓绘潈閲嶄娇鐢?`inverse_log` 椋庢牸锛堜笌鎹熷け鍑芥暟绫绘潈閲嶄繚鎸佷竴鑷达級銆?    - 瀵规甯哥被锛坈lass 10锛夊彲棰濆涔樹互涓嬮噰鏍峰洜瀛愶紝杩涗竴姝ョ紦瑙ｆ牱鏈富瀵笺€?    """

    if len(labels) == 0:
        raise ValueError("labels 涓嶈兘涓虹┖锛屾棤娉曟瀯寤?WeightedRandomSampler")

    label_array = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(label_array, minlength=CFG.num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    # Sampler-side balancing uses inverse frequency directly.
    class_weights = 1.0 / counts

    # Optional extra boost for hard classes.
    sampler_boost_map = dict(getattr(CFG, "sampler_class_boost_map", {}))
    for class_id, boost_factor in sampler_boost_map.items():
        idx = int(class_id)
        if 0 <= idx < class_weights.shape[0]:
            class_weights[idx] *= float(boost_factor)

    normal_class_id = int(CFG.fault_to_class.get(10, 10))
    normal_ratio = float(getattr(CFG, "normal_class_downsample_ratio", 1.0))
    normal_ratio = min(max(normal_ratio, 0.0), 1.0)
    class_weights[normal_class_id] *= normal_ratio
    class_weights = class_weights / class_weights.mean()

    sample_weights = class_weights[label_array]
    weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
    samples = int(num_samples) if num_samples is not None else len(labels)
    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=samples,
        replacement=bool(replacement),
    )


def build_weighted_sampler(
    records: List[Dict],
    num_samples: Optional[int] = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """
    瀵瑰鍏紑鐨勫姞鏉冮噰鏍峰櫒鏋勫缓鍑芥暟锛堟祴璇曚笌璁粌鍏变韩锛夈€?
    杈撳叆锛?    - `records`: 姣忎釜鍏冪礌闇€鍖呭惈 `class_id` 瀛楁銆?    """

    labels = [int(rec["class_id"]) for rec in records]
    return _build_weighted_sampler_from_labels(
        labels=labels,
        num_samples=num_samples,
        replacement=replacement,
    )


def build_sampler_for_train_dataset(
    train_dataset: Any,
    train_records: List[Dict],
) -> WeightedRandomSampler:
    """
    鎸夊疄闄?Dataset 绱㈠紩椤哄簭鏋勫缓閲囨牱鍣紝纭繚鏉冮噸鍜屾牱鏈竴涓€瀵瑰簲銆?    """

    dataset_obj = getattr(train_dataset, "_inner", train_dataset)

    if hasattr(dataset_obj, "stage_records"):
        stage_records = getattr(dataset_obj, "stage_records")
        labels = [int(item.class_id) for item in stage_records]
    elif hasattr(dataset_obj, "samples_per_mission"):
        samples_per_mission = int(getattr(dataset_obj, "samples_per_mission"))
        labels = []
        for rec in train_records:
            labels.extend([int(rec["class_id"])] * samples_per_mission)
    else:
        raise ValueError("鏃犳硶浠庤缁冩暟鎹泦鎺ㄦ柇閲囨牱鏍囩锛岃妫€鏌ユ暟鎹泦瀹炵幇")

    return _build_weighted_sampler_from_labels(
        labels=labels,
        num_samples=len(labels),
        replacement=True,
    )


def _load_optuna_runtime_context(
    run_dir: Path,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """
    璇诲彇 Optuna 鎴愭湰浼樺寲缁撴灉骞惰浆鎹负璁粌鍙敤瀵硅薄銆?
    杩斿洖:
        optimized_costs:
            [num_classes] 鎴愭湰鍚戦噺锛岃嫢鏈惎鐢ㄦ垨璇诲彇澶辫触鍒欎负 None銆?        optimized_hparams:
            杩囨护鍚庣殑鍙敤瓒呭弬鏁板瓧鍏革紝浠呭寘鍚綋鍓嶈缁冮摼璺敮鎸佺殑閿€?    """
    use_optuna_result = bool(getattr(CFG, "use_optuna_cost_result", False))
    result_filename = str(getattr(CFG, "optuna_cost_result_filename", "optuna_cost_result.json"))
    if not use_optuna_result:
        return None, {}

    result_path = run_dir / result_filename
    if not result_path.exists():
        print(f"[CostSensitive] Optuna result not found, skip: {result_path}")
        return None, {}

    try:
        optuna_result = load_optuna_cost_result(
            run_dir=run_dir,
            result_filename=result_filename,
        )
        optimized_costs = build_cost_tensor_from_best_costs(
            best_costs=optuna_result["best_costs"],
            class_id_to_name=CFG.class_id_to_name,
            num_classes=CFG.num_classes,
        ).to(device)
        optimized_hparams = extract_supported_hparams(optuna_result.get("best_hparams", {}))

        print(f"[CostSensitive] Loaded Optuna result: {result_path}")
        print(f"[CostSensitive] cost vector(by class_id): {optimized_costs.detach().cpu().numpy().round(4)}")
        if len(optimized_hparams) > 0:
            print(f"[CostSensitive] usable hparams: {optimized_hparams}")
        return optimized_costs, optimized_hparams
    except FileNotFoundError:
        print(f"[CostSensitive] Optuna result not found, skip: {result_path}")
        return None, {}
    except Exception as exc:
        print(f"[WARN] Failed to load Optuna result, fallback to regular training: {exc}")
        return None, {}


def resolve_loss_type(
    loss_type: str,
    use_focal_loss: bool,
    use_ldam_loss: bool,
) -> str:
    """
    瑙ｆ瀽鎹熷け绫诲瀷锛屽吋瀹?legacy 寮€鍏充笌鏂?loss_type 鏋氫妇閰嶇疆銆?
    瑙勫垯:
    1. 鏄惧紡缁欏嚭 ce/focal/ldam 鏃讹紝鐩存帴閲囩敤銆?    2. auto 妯″紡涓嬶紝鎸?legacy 浼樺厛绾? LDAM > Focal > CE銆?    """
    normalized = str(loss_type).strip().lower()
    if normalized in {"ce", "focal", "ldam"}:
        return normalized
    if bool(use_ldam_loss):
        return "ldam"
    if bool(use_focal_loss):
        return "focal"
    return "ce"


def resolve_use_class_balance(
    default_enabled: bool,
    sampler_mode: str,
) -> bool:
    """
    瑙ｆ瀽鏄惁鍚敤鎹熷け渚?class weight銆?
    璇存槑:
    - 褰?sampler_mode=weighted 鏃讹紝閲囨牱渚у凡鍋氱被鍒噸骞宠　锛?      榛樿鍏抽棴 loss 渚?class weight锛岄伩鍏嶉噸澶嶉噸鍔犳潈閫犳垚鍐茬獊銆?    """
    if not bool(default_enabled):
        return False
    if str(sampler_mode).strip().lower() == "weighted":
        return False
    return True


def build_train_criterion(
    train_records: List[Dict],
    device: torch.device,
    use_class_balance: bool,
    optimized_costs: Optional[torch.Tensor],
) -> nn.Module:
    """
    鎸夐厤缃粍鍚堟崯澶卞嚱鏁般€?
    璇存槑:
    1. 鑻ュ紑鍚?`use_ldam_loss`锛屼娇鐢?LDAM Loss锛堜紭鍏堢骇鏈€楂橈級銆?    2. 鑻ュ紑鍚?`use_adaptive_cost`锛屼紭鍏堜娇鐢?Optuna best_costs銆?    3. 鏈彁渚?Optuna 鎴愭湰鏃讹紝鍥為€€鍒?class weight 鎴栧叏1鎴愭湰銆?    4. 鍏朵粬鍒嗘敮淇濇寔鍏煎: Focal / Label Smoothing / CrossEntropy銆?    """
    class_weights: Optional[torch.Tensor] = None
    if use_class_balance:
        class_weights = _compute_class_weights(train_records).to(device)
        print(f"[Loss] class weights: {class_weights.detach().cpu().numpy().round(4)}")

    raw_loss_type = str(getattr(CFG, "loss_type", "auto"))
    loss_type = resolve_loss_type(
        loss_type=raw_loss_type,
        use_focal_loss=bool(getattr(CFG, "use_focal_loss", False)),
        use_ldam_loss=bool(getattr(CFG, "use_ldam_loss", False)),
    )
    print(f"[Loss] loss_type={loss_type}")

    if loss_type == "ldam":
        counts = np.zeros((CFG.num_classes,), dtype=np.float64)
        for rec in train_records:
            counts[int(rec["class_id"])] += 1

        weight = class_weights if use_class_balance else None
        max_m = float(getattr(CFG, "ldam_max_m", 0.5))
        scale = float(getattr(CFG, "ldam_scale", 30.0))
        print(f"[Loss] LDAMLoss(max_m={max_m}, s={scale})")
        return LDAMLoss(
            cls_num_list=counts.tolist(),
            max_m=max_m,
            weight=weight,
            s=scale,
        )

    if loss_type == "focal":
        alpha = class_weights if use_class_balance else None
        print(f"[Loss] FocalLoss(gamma={CFG.focal_gamma})")
        return FocalLoss(alpha=alpha, gamma=float(CFG.focal_gamma))

    # In explicit ce/focal/ldam modes, do not mix with AdaptiveCost.
    if str(raw_loss_type).strip().lower() == "auto" and bool(getattr(CFG, "use_adaptive_cost", False)):
        if optimized_costs is not None:
            base_costs = optimized_costs
            print("[Loss] AdaptiveCostSensitiveLoss + Optuna best_costs")
        elif class_weights is not None:
            base_costs = class_weights
            print("[Loss] AdaptiveCostSensitiveLoss + class_weights(fallback)")
        else:
            base_costs = torch.ones(CFG.num_classes, dtype=torch.float32, device=device)
            print("[Loss] AdaptiveCostSensitiveLoss + uniform costs(fallback)")

        label_smoothing = float(CFG.label_smoothing) if bool(CFG.use_label_smoothing) else 0.0
        return AdaptiveCostSensitiveLoss(
            num_classes=CFG.num_classes,
            base_costs=base_costs,
            label_smoothing=label_smoothing,
        )

    if bool(getattr(CFG, "use_label_smoothing", False)):
        weight = class_weights if use_class_balance else None
        print(f"[Loss] LabelSmoothingCrossEntropy(smoothing={CFG.label_smoothing})")
        return LabelSmoothingCrossEntropy(
            smoothing=float(CFG.label_smoothing),
            weight=weight,
        )

    if use_class_balance:
        print("[Loss] CrossEntropyLoss + class_weights")
        return nn.CrossEntropyLoss(weight=class_weights)

    print("[Loss] CrossEntropyLoss")
    return nn.CrossEntropyLoss()


def train_source_model(
    run_dir: Path,
    model_name: str = "tcn_source",
    use_class_balance: bool = True,
    max_epochs: Optional[int] = None,
    model_class_name: str = "temporal_convnet",
) -> TrainResult:
    """
    璁粌婧愬煙妯″瀷
    
    鍙傛暟:
        run_dir: 杩愯鐩綍
        model_name: 妯″瀷鍚嶇О锛堢敤浜?checkpoint 鍛藉悕锛?        use_class_balance: 鏄惁浣跨敤绫诲埆骞宠　鏉冮噸
        max_epochs: 鏈€澶ц缁冭疆鏁帮紙None 鍒欎娇鐢ㄩ厤缃級
        model_class_name: 妯″瀷绫诲悕绉帮紙榛樿 temporal_convnet锛?    
    杩斿洖:
        TrainResult
    """
    set_global_seed(CFG.run_seed)
    device = resolve_device(CFG.device)
    parallel_trials = int(os.getenv("UAV_TCN_PARALLEL_TRIALS", "1"))
    num_workers = resolve_num_workers(CFG.num_workers, parallel_trials=parallel_trials)
    max_epochs = max_epochs if max_epochs is not None else CFG.max_epochs
    apply_optuna_best_hparams = bool(getattr(CFG, "apply_optuna_best_hparams", False))
    
    # 鍔犺浇鏁版嵁
    train_records, val_records, mean, std = _load_source_artifacts(run_dir)
    
    # Prefer precomputed samples when available.
    precomputed_train_dir = run_dir / "precomputed" / "train"
    precomputed_val_dir = run_dir / "precomputed" / "val"
    use_precomputed_train = (
        precomputed_train_dir.exists()
        and (precomputed_train_dir / "manifest.json").exists()
    )
    use_precomputed_val = (
        precomputed_val_dir.exists()
        and (precomputed_val_dir / "manifest.json").exists()
    )

    if use_precomputed_train:
        print("[数据] 使用预计算训练样本（离线缓存加速）")
        train_dataset = PrecomputedDataset(
            precomputed_dir=precomputed_train_dir,
            is_train=True,
            base_seed=CFG.run_seed,
        )
    else:
        print("[数据] 使用在线计算模式")
        loader = MissionLoader(max_cache_items=128)
        train_dataset = SourceDomainDataset(
            records=train_records,
            zscore_mean=mean,
            zscore_std=std,
            loader=loader,
            is_train=True,
            base_seed=CFG.run_seed,
            enable_targeted_gan=False,
        )

    if use_precomputed_val:
        print("[数据] 使用预计算验证样本（离线缓存加速）")
        val_dataset = PrecomputedDataset(
            precomputed_dir=precomputed_val_dir,
            is_train=False,
            base_seed=CFG.run_seed,
        )
    else:
        if not use_precomputed_train:
            val_loader_obj = loader  # 澶嶇敤宸插垱寤虹殑 loader
        else:
            val_loader_obj = MissionLoader(max_cache_items=128)
        val_dataset = DeterministicWindowDataset(
            records=val_records,
            zscore_mean=mean,
            zscore_std=std,
            loader=val_loader_obj,
            windows_per_scale=CFG.eval_windows_per_scale,
            base_seed=CFG.run_seed,
        )

    dataloader_runtime_kwargs = build_dataloader_runtime_kwargs(
        num_workers=num_workers,
        device=device,
        pin_memory=bool(CFG.pin_memory),
        persistent_workers=bool(CFG.persistent_workers),
        prefetch_factor=int(CFG.prefetch_factor),
    )
    if bool(dataloader_runtime_kwargs.get("persistent_workers", False)):
        # Disable persistent workers for train-time dynamic augmentation state.
        dataloader_runtime_kwargs.pop("persistent_workers", None)
        print("[DataLoader] train: persistent_workers disabled to avoid stale worker state")

    train_sampler = None
    train_shuffle = True
    if str(getattr(CFG, "sampler_mode", "none")).lower() == "weighted":
        train_sampler = build_sampler_for_train_dataset(
            train_records=train_records,
            train_dataset=train_dataset,
        )
        train_shuffle = False
        sampler_boost_map = dict(getattr(CFG, "sampler_class_boost_map", {}))
        if len(sampler_boost_map) > 0:
            print(
                f"[采样器] 使用 WeightedRandomSampler（按类别加权采样），"
                f"hard_class_boost={sampler_boost_map}"
            )
        else:
            print("[Sampler] Using WeightedRandomSampler")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_padded,
        drop_last=True,
        **dataloader_runtime_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_padded,
        drop_last=False,
        **dataloader_runtime_kwargs,
    )
    
    # Create model and load optional Optuna runtime context.
    optimized_costs, optimized_hparams = _load_optuna_runtime_context(run_dir, device)

    # 璁粌鏈熷彲閫夊簲鐢?Optuna 鏈€浼樿秴鍙傛暟
    effective_lr = float(CFG.learning_rate)
    model_kwargs: Dict[str, Any] = {}
    if apply_optuna_best_hparams and len(optimized_hparams) > 0:
        if "learning_rate" in optimized_hparams:
            effective_lr = float(optimized_hparams["learning_rate"])
        for key in ("tcn_kernel_size", "tcn_channels", "tcn_dropout"):
            if key in optimized_hparams:
                model_kwargs[key] = optimized_hparams[key]
        print(f"[Train HParams] apply Optuna best hparams | lr={effective_lr}, model_kwargs={model_kwargs}")
    elif apply_optuna_best_hparams:
        print("[WARN] apply_optuna_best_hparams enabled but no usable best_hparams found; fallback to config defaults.")

    resolved_model_class_name = normalize_model_class_name(model_class_name)
    model_init_kwargs = build_model_init_kwargs_from_cfg(
        overrides=model_kwargs,
        model_class_name=resolved_model_class_name,
    )
    model_class = resolve_model_class(resolved_model_class_name)
    model = model_class(**model_init_kwargs).to(device)
    print(f"[{model_name}] model_class={resolved_model_class_name}")
    # 鏂规 A+E锛欰damW锛堣В鑰?weight decay锛? CosineAnnealingWarmRestarts
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=float(getattr(CFG, 'weight_decay', 0.0)),
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp_enabled = bool(CFG.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    non_blocking = bool(device.type == "cuda" and CFG.pin_memory)
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    
    # 鎹熷け鍑芥暟
    effective_use_class_balance = resolve_use_class_balance(
        default_enabled=bool(use_class_balance),
        sampler_mode=str(getattr(CFG, "sampler_mode", "none")),
    )
    if bool(use_class_balance) and not effective_use_class_balance:
        print("[Imbalance] sampler_mode=weighted, disable loss-side class_weight to avoid double weighting")

    criterion = build_train_criterion(
        train_records=train_records,
        device=device,
        use_class_balance=effective_use_class_balance,
        optimized_costs=optimized_costs,
    )
    
    # Exp-A: 混淆感知损失 + SupCon/CenterLoss
    supcon_loss = None
    supcon_lambda = 0.0
    center_loss = None
    center_lambda = 0.0
    if bool(getattr(CFG, "use_confusion_aware_loss", False)):
        feature_dim = int(getattr(model.classifier, "in_features", CFG.tcn_channels))
        criterion, supcon_loss, supcon_lambda, center_loss, center_lambda = build_exp_a_criterion(
            train_records=train_records,
            device=device,
            use_class_balance=effective_use_class_balance,
            feature_dim=feature_dim,
        )
        print(
            f"[Exp-A] 启用混淆感知损失 + 辅助特征损失 "
            f"(supcon_lambda={supcon_lambda}, center_lambda={center_lambda})"
        )
    center_optimizer = None
    if center_loss is not None:
        center_lr = float(getattr(CFG, "center_loss_lr", 0.5))
        center_optimizer = torch.optim.SGD(center_loss.parameters(), lr=center_lr)
        print(f"[Exp-A] CenterLoss optimizer: SGD(lr={center_lr})")
    
    # 创建日志与 checkpoint 目录
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{model_name}.jsonl"
    log_fp = (
        open(log_path, "w", encoding="utf-8")
        if bool(getattr(CFG, "log_write_jsonl", False))
        else None
    )

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{model_name}.pt"

    def _log(event: Dict[str, Any]) -> None:
        if log_fp is not None:
            event["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            log_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
            log_fp.flush()

    # 训练状态初始化
    best_val_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []
    
    # 鏋勫缓 LR Scheduler
    scheduler = None
    sched_type = str(getattr(CFG, 'scheduler_type', 'none')).lower()
    if sched_type == 'cosine_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(getattr(CFG, 'scheduler_T_0', 5)),
            T_mult=int(getattr(CFG, 'scheduler_T_mult', 2)),
            eta_min=float(getattr(CFG, 'scheduler_eta_min', 1e-6)),
        )
        print(f"[{model_name}] LR Scheduler: CosineAnnealingWarmRestarts "
              f"T_0={CFG.scheduler_T_0} T_mult={CFG.scheduler_T_mult} eta_min={CFG.scheduler_eta_min}")
    else:
        print(f"[{model_name}] LR Scheduler: disabled")

    t_start = time.perf_counter()
    val_interval_epochs = max(1, int(getattr(CFG, "val_interval_epochs", 1)))
    fail_on_nan = bool(getattr(CFG, "fail_on_nan", True))
    max_invalid_batches = 3
    stop_due_to_nan = False

    for epoch in range(max_epochs):
        # 鏇存柊鏁版嵁闆嗙殑 epoch
        train_dataset.set_epoch(epoch)

        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        invalid_batches = 0

        for batch_idx, (x, lengths, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=non_blocking)
            lengths = lengths.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            if center_optimizer is not None:
                center_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=autocast_device_type,
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ):
                # 缁村害璺熻釜:
                # x: [B, T, C] -> logits: [B, num_classes]
                # Exp-A: 支持 SupConLoss/CenterLoss（需要 pooled 特征）
                need_aux_features = (supcon_loss is not None) or (center_loss is not None)
                pooled = None
                if need_aux_features:
                    logits, aux = model.forward_with_aux(x, lengths)
                    pooled_raw = aux.get("pooled_features")
                    if isinstance(pooled_raw, torch.Tensor):
                        pooled = pooled_raw
                    else:
                        pooled = model.get_features(x, lengths)
                else:
                    logits = model(x, lengths)
                loss = criterion(logits, y)
                if supcon_loss is not None and pooled is not None:
                    supcon_val = supcon_loss(pooled.float(), y)
                    loss = loss + supcon_lambda * supcon_val
                if center_loss is not None and pooled is not None:
                    center_val = center_loss(pooled.float(), y)
                    loss = loss + center_lambda * center_val
                if hasattr(model, "get_additional_regularization_loss"):
                    reg_loss_raw = model.get_additional_regularization_loss()
                    if isinstance(reg_loss_raw, torch.Tensor):
                        reg_loss = (
                            reg_loss_raw
                            if reg_loss_raw.ndim == 0
                            else reg_loss_raw.mean()
                        )
                        if torch.isfinite(reg_loss):
                            loss = loss + reg_loss

            if (not torch.isfinite(logits).all()) or (not torch.isfinite(loss)):
                invalid_batches += 1
                print(
                    f"[WARN] {model_name} epoch={epoch+1} batch={batch_idx} 出现非有限值，已跳过更新 "
                    f"(invalid_batches={invalid_batches})"
                )
                _log(
                    {
                        "event": "invalid_batch",
                        "epoch": epoch,
                        "batch": batch_idx,
                        "invalid_batches": invalid_batches,
                    }
                )
                optimizer.zero_grad(set_to_none=True)
                if fail_on_nan and invalid_batches >= max_invalid_batches:
                    stop_due_to_nan = True
                    print(
                        f"[{model_name}] consecutive invalid batches reached threshold ({max_invalid_batches}); stop training early."
                    )
                    break
                continue

            invalid_batches = 0
            grad_clip_norm = float(getattr(CFG, "grad_clip_norm", 0.0))
            if amp_enabled:
                scaler.scale(loss).backward()
                if grad_clip_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if center_optimizer is not None:
                    scaler.unscale_(center_optimizer)
                    if center_lambda > 0.0:
                        for param in center_loss.parameters():  # type: ignore[union-attr]
                            if param.grad is not None:
                                param.grad.mul_(1.0 / max(center_lambda, 1e-12))
                scaler.step(optimizer)
                if center_optimizer is not None:
                    scaler.step(center_optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                if center_optimizer is not None:
                    if center_lambda > 0.0:
                        for param in center_loss.parameters():  # type: ignore[union-attr]
                            if param.grad is not None:
                                param.grad.mul_(1.0 / max(center_lambda, 1e-12))
                    center_optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1


        if stop_due_to_nan:
            break

        train_loss = train_loss_sum / max(train_batches, 1)
        should_validate = (
            epoch == 0
            or ((epoch + 1) % val_interval_epochs == 0)
            or (epoch + 1 == max_epochs)
        )
        val_macro_f1: float | None = None

        if should_validate:
            # 楠岃瘉
            model.eval()
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for x, lengths, y in val_loader:
                    x = x.to(device, non_blocking=non_blocking)
                    lengths = lengths.to(device, non_blocking=non_blocking)
                    with torch.amp.autocast(
                        device_type=autocast_device_type,
                        dtype=autocast_dtype,
                        enabled=amp_enabled,
                    ):
                        # 缁村害璺熻釜:
                        # x: [B, T, C] -> logits: [B, num_classes]
                        logits = model(x, lengths)
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    val_preds.extend(preds.tolist())
                    val_labels.extend(y.numpy().tolist())

            val_macro_f1 = float(f1_score(val_labels, val_preds, average="macro"))

            # Track best checkpoint by val macro-F1.
            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                best_epoch = epoch
                epochs_no_improve = 0

                # 淇濆瓨 checkpoint
                torch.save(
                    {
                        "model_name": model_name,
                        "model_class_name": resolved_model_class_name,
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "val_macro_f1": val_macro_f1,
                        "optimized_hparams": optimized_hparams,
                        "optimized_costs": None
                        if optimized_costs is None
                        else optimized_costs.detach().cpu().tolist(),
                        "model_init_kwargs": model_init_kwargs,
                    },
                    ckpt_path,
                )
            else:
                epochs_no_improve += 1

        elapsed = time.perf_counter() - t_start
        if val_macro_f1 is None:
            print(
                f"[{model_name}] epoch={epoch+1} train_loss={train_loss:.5f} "
                f"val=skip(interval={val_interval_epochs}) elapsed={elapsed:.0f}s"
            )
        else:
            print(
                f"[{model_name}] epoch={epoch+1} train_loss={train_loss:.5f} "
                f"val_f1={val_macro_f1:.4f} elapsed={elapsed:.0f}s"
            )

        _log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_macro_f1": val_macro_f1,
                "val_computed": bool(should_validate),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": round(elapsed, 2),
            }
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_macro_f1": val_macro_f1,
                "val_computed": bool(should_validate),
            }
        )

        # Early Stopping
        if should_validate and epochs_no_improve >= CFG.early_stopping_patience:
            print(f"[{model_name}] Early stopping at epoch {epoch+1}")
            break

        # Epoch 缁撴潫鍚庢洿鏂?LR Scheduler
        if scheduler is not None:
            scheduler.step()
    
    if log_fp is not None:
        log_fp.close()
    
    if best_epoch >= 0:
        print(f"[{model_name}] 训练完成 | best_epoch={best_epoch+1} best_val_f1={best_val_f1:.4f}")
    else:
        print(f"[{model_name}] 训练完成，但未获得有效验证结果（best checkpoint 未更新）")
    
    return TrainResult(
        model_name=model_name,
        best_epoch=best_epoch,
        best_val_macro_f1=float(best_val_f1),
        checkpoint_path=str(ckpt_path),
        history=history,
    )



def _resolve_center_loss_active_class_ids() -> tuple[int, ...]:
    """
    解析 CenterLoss 生效类别。
    优先使用显式配置；未配置时回退到 confusion_pairs 覆盖类集合。
    """
    explicit_ids = tuple(int(v) for v in getattr(CFG, "center_loss_active_classes", ()))
    if len(explicit_ids) > 0:
        return explicit_ids

    pairs = tuple(getattr(CFG, "confusion_pairs", ()))
    inferred_ids: List[int] = []
    for pair in pairs:
        if len(pair) >= 2:
            inferred_ids.append(int(pair[0]))
            inferred_ids.append(int(pair[1]))
    return tuple(sorted(set(inferred_ids)))


# ============ Exp-A: 混淆感知损失配置 ============
def build_exp_a_criterion(
    train_records: List[Dict],
    device: torch.device,
    use_class_balance: bool,
    feature_dim: int,
) -> Tuple[nn.Module, Optional[SupConLoss], float, Optional[CenterLoss], float]:
    """
    构建 Exp-A 损失函数组合：ConfusionAwareLoss + SupConLoss
    
    配置项（在 config.yaml 中）：
    - use_confusion_aware_loss: bool
    - confusion_pairs: list of [src, dst, scale]
    - lambda_confusion: float
    - use_supcon_loss: bool
    - supcon_temperature: float
    - supcon_lambda: float
    
    返回:
        (classification_loss, supcon_loss, supcon_lambda, center_loss, center_lambda)
    """
    # 计算类别权重
    class_weights = None
    if use_class_balance:
        class_weights = _compute_class_weights(train_records).to(device)
        print(f"[Exp-A] class weights: {class_weights.detach().cpu().numpy().round(4)}")
    
    # FocalLoss 作为基础损失
    focal_gamma = float(getattr(CFG, "focal_gamma", 2.0))
    base_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma, reduction="none")
    print(f"[Exp-A] Base loss: FocalLoss(gamma={focal_gamma})")
    
    # ConfusionAwareLoss
    confusion_pairs = getattr(CFG, "confusion_pairs", None)
    if confusion_pairs is None:
        # 默认混淆对（基于混淆矩阵分析）
        confusion_pairs = [
            [8, 10, 3.0],  # barometer -> low_voltage
            [8, 9, 2.5],   # barometer -> GPS
            [10, 8, 2.5],  # low_voltage -> barometer
            [9, 0, 2.0],   # GPS -> motor
            [0, 9, 2.0],   # motor -> GPS
        ]
    lambda_confusion = float(getattr(CFG, "lambda_confusion", 1.0))
    
    # 转换为 tuple list
    confusion_pairs_tuples = [tuple(p) for p in confusion_pairs]
    
    criterion = ConfusionAwareLoss(
        base_loss=base_loss,
        confusion_pairs=confusion_pairs_tuples,
        lambda_confusion=lambda_confusion,
    )
    print(f"[Exp-A] ConfusionAwareLoss with {len(confusion_pairs)} pairs, lambda={lambda_confusion}")
    
    # SupConLoss（可选）
    supcon_loss = None
    supcon_lambda = 0.0
    if bool(getattr(CFG, "use_supcon_loss", False)):
        supcon_temperature = float(getattr(CFG, "supcon_temperature", 0.1))
        supcon_lambda = float(getattr(CFG, "supcon_lambda", 0.3))
        supcon_loss = SupConLoss(
            temperature=supcon_temperature,
            proj_head=None,
            normalize=True,
        )
        print(f"[Exp-A] SupConLoss(temperature={supcon_temperature}, lambda={supcon_lambda})")

    # CenterLoss（可选）
    center_loss = None
    center_lambda = 0.0
    if bool(getattr(CFG, "use_center_loss", False)):
        center_lambda = float(getattr(CFG, "center_loss_lambda", 0.1))
        active_class_ids = _resolve_center_loss_active_class_ids()
        center_loss = CenterLoss(
            num_classes=int(CFG.num_classes),
            feat_dim=int(feature_dim),
            active_class_ids=list(active_class_ids),
        ).to(device)
        print(
            f"[Exp-A] CenterLoss(feat_dim={feature_dim}, lambda={center_lambda}, "
            f"active_classes={list(active_class_ids)})"
        )

    return criterion, supcon_loss, supcon_lambda, center_loss, center_lambda
