"""
fine_tune.py

鐩爣鍩熷井璋冮€昏緫

鑱岃矗:
1. 鍔犺浇婧愬煙棰勮缁冩ā鍨?2. 鍦ㄧ洰鏍囧煙鏁版嵁涓婂井璋?3. 鏀寔鍏ㄩ噺寰皟/鍐荤粨鐗瑰緛鎻愬彇鍣?"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

from essence_forge.core.runtime_config import CFG
from essence_forge.core.datasets import (
    TargetDomainDataset,
    DeterministicWindowDataset,
    collate_padded,
)
from essence_forge.core.fine_tune_metrics import (
    compute_early_stopping_score,
    compute_validation_metrics,
)
from essence_forge.core.precomputed_dataset import PrecomputedDataset
from essence_forge.core.rflymad_io import MissionLoader
from essence_forge.core.model_checkpoint import load_temporal_convnet_from_checkpoint
from essence_forge.core.train import (
    build_exp_a_criterion,
    build_sampler_for_train_dataset,
    build_train_criterion,
    resolve_loss_type,
    resolve_use_class_balance,
)
from essence_forge.core.utils import (
    build_dataloader_runtime_kwargs,
    read_json,
    resolve_device,
    resolve_num_workers,
    set_global_seed,
)


@dataclass
class FineTuneResult:
    """寰皟缁撴灉璁板綍"""
    model_name: str
    best_epoch: int
    best_val_macro_f1: float
    best_val_gmean: float
    checkpoint_path: str
    history: List[Dict]


def _load_target_artifacts(run_dir: Path):
    """Load target-domain splits and source normalization statistics."""
    train_path = run_dir / "split_target_train.json"
    val_path = run_dir / "split_target_val.json"
    
    # Use source-domain statistics for normalization.
    stats_path = run_dir / "source_stats.json"
    
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Missing split_target_*.json. Please run index first.")
    if not stats_path.exists():
        raise FileNotFoundError("Missing source_stats.json. Please run index first.")
    
    train_records = read_json(train_path)
    val_records = read_json(val_path)
    stats = read_json(stats_path)
    
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    
    return train_records, val_records, mean, std


def _build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    model_name: str,
    epoch: int,
    val_macro_f1: float,
    val_gmean: float,
    monitor_score: float,
    monitor_metric: str,
    source_checkpoint: str,
    load_meta: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model_name": model_name,
        "model_class_name": load_meta.get("model_class_name", "temporal_convnet"),
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "val_macro_f1": float(val_macro_f1),
        "val_gmean": float(val_gmean),
        "monitor_score": float(monitor_score),
        "monitor_metric": str(monitor_metric),
        "source_checkpoint": source_checkpoint,
        "model_init_kwargs": load_meta["model_init_kwargs"],
    }
    if isinstance(extra, dict):
        payload.update(extra)
    return payload


def _average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(state_dicts) == 0:
        raise ValueError("state_dicts is empty")
    if len(state_dicts) == 1:
        return {
            k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in state_dicts[0].items()
        }

    averaged: Dict[str, torch.Tensor] = {}
    first = state_dicts[0]
    for key, base_tensor in first.items():
        if not isinstance(base_tensor, torch.Tensor):
            continue
        tensors = [state[key].detach().cpu() for state in state_dicts]
        if torch.is_floating_point(base_tensor):
            stacked = torch.stack([t.to(dtype=torch.float32) for t in tensors], dim=0)
            mean_tensor = stacked.mean(dim=0).to(dtype=base_tensor.dtype)
            averaged[key] = mean_tensor
        else:
            averaged[key] = tensors[0].clone()
    return averaged


def fine_tune_model(
    run_dir: Path,
    source_checkpoint: str,
    model_name: str = "tcn_finetuned",
    freeze_layers: Optional[int] = None,
    learning_rate: Optional[float] = None,
    max_epochs: Optional[int] = None,
) -> FineTuneResult:
    """
    Fine-tune source model on target-domain data.
    """
    set_global_seed(CFG.run_seed)
    device = resolve_device(CFG.device)
    parallel_trials = int(os.getenv("UAV_TCN_PARALLEL_TRIALS", "1"))
    num_workers = resolve_num_workers(CFG.num_workers, parallel_trials=parallel_trials)
    
    freeze_layers = freeze_layers if freeze_layers is not None else CFG.ft_freeze_layers
    learning_rate = learning_rate if learning_rate is not None else CFG.ft_learning_rate
    lr_backbone = float(getattr(CFG, "ft_lr_backbone", learning_rate))
    lr_head = float(getattr(CFG, "ft_lr_head", learning_rate))
    freeze_epochs = int(getattr(CFG, "ft_freeze_epochs", 0))
    max_epochs = max_epochs if max_epochs is not None else CFG.ft_max_epochs
    
    # Load target-domain artifacts.
    train_records, val_records, mean, std = _load_target_artifacts(run_dir)
    
    # Prefer precomputed samples when available.
    precomputed_target_train = run_dir / "precomputed" / "target_train"
    precomputed_target_val = run_dir / "precomputed" / "target_val"
    use_precomputed_train = (
        precomputed_target_train.exists()
        and (precomputed_target_train / "manifest.json").exists()
    )
    use_precomputed_val = (
        precomputed_target_val.exists()
        and (precomputed_target_val / "manifest.json").exists()
    )

    if use_precomputed_train:
        print("[数据] 使用预计算目标域训练样本（离线缓存加速）")
        train_dataset = PrecomputedDataset(
            precomputed_dir=precomputed_target_train,
            is_train=True,
            base_seed=CFG.run_seed,
            augment_mode=str(getattr(CFG, "ft_augment_mode", "full")),
        )
    else:
        print("[数据] 使用在线计算模式")
        loader = MissionLoader(max_cache_items=128)
        train_dataset = TargetDomainDataset(
            records=train_records,
            zscore_mean=mean,
            zscore_std=std,
            loader=loader,
            is_train=True,
            base_seed=CFG.run_seed,
            enable_targeted_gan=bool(getattr(CFG, "targeted_gan_enable", False)),
            augment_mode=str(getattr(CFG, "ft_augment_mode", "full")),
        )

    if use_precomputed_val:
        print("[数据] 使用预计算目标域验证样本（离线缓存加速）")
        val_dataset = PrecomputedDataset(
            precomputed_dir=precomputed_target_val,
            is_train=False,
            base_seed=CFG.run_seed,
        )
    else:
        if not use_precomputed_train:
            val_loader_obj = loader
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
        # Disable persistent workers for epoch-dependent train-time augmentation state.
        dataloader_runtime_kwargs.pop("persistent_workers", None)
        print("[DataLoader] finetune: persistent_workers disabled to avoid stale worker state")

    train_sampler = None
    train_shuffle = True
    if str(getattr(CFG, "sampler_mode", "none")).lower() == "weighted":
        train_sampler = build_sampler_for_train_dataset(
            train_dataset=train_dataset,
            train_records=train_records,
        )
        train_shuffle = False
        sampler_boost_map = dict(getattr(CFG, "sampler_class_boost_map", {}))
        if len(sampler_boost_map) > 0:
            print(
                f"[Sampler] finetune uses WeightedRandomSampler, "
                f"hard_class_boost={sampler_boost_map}"
            )
        else:
            print("[Sampler] finetune uses WeightedRandomSampler")

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
    
    # 加载预训练模型
    print(f"[{model_name}] 加载预训练模型: {source_checkpoint}")
    model, ckpt, load_meta = load_temporal_convnet_from_checkpoint(
        checkpoint=source_checkpoint,
        device=device,
    )
    if load_meta["model_kwargs_source"] == "inferred_from_state_dict":
        print(
            "[WARN] checkpoint lacks model_init_kwargs; loaded model structure inferred from state_dict."
        )
    
    print(f"  源域模型 | epoch={ckpt['epoch']} val_f1={ckpt.get('val_macro_f1', 'N/A')}")
    
    # 寰皟绛栫暐锛氬垎灞傚涔犵巼 + 鍓?N epoch 鍐荤粨 backbone锛堝吋瀹?freeze_layers 鏃ц涔夛級
    freeze_full_training = freeze_layers > 0 and freeze_epochs <= 0
    should_freeze_backbone = freeze_full_training or freeze_epochs > 0
    if should_freeze_backbone:
        model.freeze_feature_extractor(freeze=True)
    else:
        model.freeze_feature_extractor(freeze=False)

    # 前端冻结：仅冻结 LWPT frontend + freq_projection，保留 TCN backbone 可训练
    # 用途：保护源域学到的残差通道频率分解模式
    freeze_frontend = bool(getattr(CFG, "ft_freeze_frontend", False))
    if freeze_frontend and hasattr(model, "freeze_frontend"):
        model.freeze_frontend(freeze=True)

    if not hasattr(model, "classifier"):
        raise AttributeError("模型缺少 classifier，无法构建 head/backbone 分层学习率参数组")
    head_params = list(model.classifier.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids and p.requires_grad]
    if len(head_params) == 0 or len(backbone_params) == 0:
        raise ValueError("Layer-wise LR parameter groups are empty; please check model parameter grouping.")

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
            {"params": head_params, "lr": lr_head, "name": "head"},
        ],
        weight_decay=float(getattr(CFG, 'ft_weight_decay', 0.0)),
    )
    print(
        f"[{model_name}] 微调策略 | lr_backbone={lr_backbone:.6g} lr_head={lr_head:.6g} "
        f"freeze_epochs={freeze_epochs} freeze_layers={freeze_layers} "
        f"freeze_frontend={freeze_frontend} "
        f"initial_backbone_frozen={should_freeze_backbone}"
    )
    if freeze_full_training:
        print(f"[{model_name}] compatibility mode: freeze_layers>0 and freeze_epochs=0, backbone stays frozen.")
    effective_use_class_balance = resolve_use_class_balance(
        default_enabled=True,
        sampler_mode=str(getattr(CFG, "sampler_mode", "none")),
    )
    if not effective_use_class_balance:
        print("[Imbalance] sampler_mode=weighted, disable loss-side class_weight to avoid double weighting.")
    loss_type = resolve_loss_type(
        loss_type=str(getattr(CFG, "loss_type", "auto")),
        use_focal_loss=bool(getattr(CFG, "use_focal_loss", False)),
        use_ldam_loss=bool(getattr(CFG, "use_ldam_loss", False)),
    )
    print(f"[{model_name}] 损失策略 | loss_type={loss_type}")
    criterion = build_train_criterion(
        train_records=train_records,
        device=device,
        use_class_balance=effective_use_class_balance,
        optimized_costs=None,
    )
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
            f"[{model_name}] Exp-A 启用 | supcon_lambda={supcon_lambda} "
            f"center_lambda={center_lambda}"
        )
    center_optimizer = None
    if center_loss is not None:
        center_lr = float(getattr(CFG, "center_loss_lr", 0.5))
        center_optimizer = torch.optim.SGD(center_loss.parameters(), lr=center_lr)
        print(f"[{model_name}] CenterLoss optimizer: SGD(lr={center_lr})")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp_enabled = bool(CFG.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    non_blocking = bool(device.type == "cuda" and CFG.pin_memory)
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    
    # 鏃ュ織鍜?checkpoint
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{model_name}.jsonl"
    log_fp = open(log_path, "w", encoding="utf-8") if CFG.log_write_jsonl else None
    
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{model_name}.pt"
    
    def _log(event: Dict) -> None:
        if log_fp is not None:
            event["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            log_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
            log_fp.flush()
    
    # 璁粌寰幆
    best_val_f1 = -1.0
    best_val_gmean = 0.0
    best_monitor_score = float("-inf")
    best_epoch = -1
    epochs_no_improve = 0
    history: List[Dict] = []
    top_checkpoints: List[Dict[str, Any]] = []

    ft_grad_clip_norm = float(getattr(CFG, 'ft_grad_clip_norm', 0.0))
    early_stopping_metric = str(
        getattr(CFG, "ft_early_stopping_metric", "macro_f1")
    ).strip().lower()
    early_stopping_warmup_epochs = int(
        getattr(CFG, "ft_early_stopping_warmup_epochs", 8)
    )
    checkpoint_top_k = max(1, int(getattr(CFG, "checkpoint_top_k", 3)))
    last_checkpoint_averaging = bool(
        getattr(CFG, "last_checkpoint_averaging", False)
    )
    print(
        f"[{model_name}] 开始微调 | amp={amp_enabled} max_epochs={max_epochs} grad_clip={ft_grad_clip_norm} "
        f"early_stop_metric={early_stopping_metric} warmup_epochs={early_stopping_warmup_epochs} "
        f"checkpoint_top_k={checkpoint_top_k} last_checkpoint_averaging={last_checkpoint_averaging}"
    )

    # 鏋勫缓 LR Scheduler锛堝井璋冮樁娈碉級
    scheduler = None
    ft_sched_type = str(getattr(CFG, 'ft_scheduler_type', 'none')).lower()
    if ft_sched_type == 'cosine_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(getattr(CFG, 'ft_scheduler_T_0', 3)),
            T_mult=int(getattr(CFG, 'ft_scheduler_T_mult', 2)),
            eta_min=float(getattr(CFG, 'ft_scheduler_eta_min', 1e-6)),
        )
        print(f"[{model_name}] LR Scheduler: CosineAnnealingWarmRestarts "
              f"T_0={CFG.ft_scheduler_T_0} T_mult={CFG.ft_scheduler_T_mult} eta_min={CFG.ft_scheduler_eta_min}")
    else:
        print(f"[{model_name}] LR Scheduler: disabled")

    t_start = time.perf_counter()
    val_interval_epochs = max(1, int(getattr(CFG, "ft_val_interval_epochs", 1)))
    fail_on_nan = bool(getattr(CFG, "ft_fail_on_nan", True))
    max_invalid_batches = 3
    stop_due_to_nan = False

    for epoch in range(max_epochs):
        train_dataset.set_epoch(epoch)

        if freeze_epochs > 0 and epoch == freeze_epochs and not freeze_full_training:
            model.freeze_feature_extractor(freeze=False)
            print(f"[{model_name}] epoch={epoch+1} unfreeze backbone and switch to full fine-tuning")

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
                    loss = loss + supcon_lambda * supcon_loss(pooled.float(), y)
                if center_loss is not None and pooled is not None:
                    loss = loss + center_lambda * center_loss(pooled.float(), y)
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
                    f"[WARN] {model_name} epoch={epoch+1} batch={batch_idx} 鍑虹幇闈炴湁闄愬€硷紝宸茶烦杩囨洿鏂?"
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
                        f"[{model_name}] consecutive invalid batches reached threshold ({max_invalid_batches}); stop fine-tuning early."
                    )
                    break
                continue

            invalid_batches = 0
            ft_grad_clip_norm = float(getattr(CFG, "ft_grad_clip_norm", 0.0))
            if amp_enabled:
                scaler.scale(loss).backward()
                if ft_grad_clip_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), ft_grad_clip_norm)
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
                if ft_grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), ft_grad_clip_norm)
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
        val_gmean: float | None = None
        monitor_score: float | None = None

        if should_validate:
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
                        logits = model(x, lengths)
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    val_preds.extend(preds.tolist())
                    val_labels.extend(y.numpy().tolist())

            metrics = compute_validation_metrics(y_true=val_labels, y_pred=val_preds)
            val_macro_f1 = float(metrics["val_macro_f1"])
            val_gmean = float(metrics["val_gmean"])
            monitor_score = compute_early_stopping_score(
                metrics=metrics,
                metric_name=early_stopping_metric,
            )

            ckpt_payload = _build_checkpoint_payload(
                model=model,
                model_name=model_name,
                epoch=epoch,
                val_macro_f1=val_macro_f1,
                val_gmean=val_gmean,
                monitor_score=monitor_score,
                monitor_metric=early_stopping_metric,
                source_checkpoint=source_checkpoint,
                load_meta=load_meta,
            )
            top_ckpt_path = ckpt_dir / f"{model_name}.topk_epoch{epoch + 1:03d}.pt"
            torch.save(ckpt_payload, top_ckpt_path)
            top_checkpoints.append(
                {
                    "path": top_ckpt_path,
                    "epoch": int(epoch),
                    "monitor_score": float(monitor_score),
                    "val_macro_f1": float(val_macro_f1),
                    "val_gmean": float(val_gmean),
                }
            )
            top_checkpoints.sort(
                key=lambda item: (
                    float(item["monitor_score"]),
                    float(item["val_macro_f1"]),
                    float(item["val_gmean"]),
                ),
                reverse=True,
            )
            while len(top_checkpoints) > checkpoint_top_k:
                removed = top_checkpoints.pop(-1)
                removed_path = Path(str(removed["path"]))
                if removed_path.exists():
                    removed_path.unlink()

            improved = float(monitor_score) > float(best_monitor_score) + 1e-12
            if improved:
                best_monitor_score = float(monitor_score)
                best_val_f1 = val_macro_f1
                best_val_gmean = val_gmean
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(ckpt_payload, ckpt_path)
            elif (epoch + 1) > early_stopping_warmup_epochs:
                epochs_no_improve += 1

        elapsed = time.perf_counter() - t_start
        if val_macro_f1 is None or val_gmean is None:
            print(
                f"[{model_name}] epoch={epoch+1} train_loss={train_loss:.5f} "
                f"val=skip(interval={val_interval_epochs}) elapsed={elapsed:.0f}s"
            )
        else:
            score_text = (
                "N/A"
                if monitor_score is None
                else f"{float(monitor_score):.4f}"
            )
            print(
                f"[{model_name}] epoch={epoch+1} train_loss={train_loss:.5f} "
                f"val_f1={val_macro_f1:.4f} val_gmean={val_gmean:.4f} "
                f"monitor({early_stopping_metric})={score_text} elapsed={elapsed:.0f}s"
            )

        _log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_macro_f1": val_macro_f1,
                "val_gmean": val_gmean,
                "monitor_metric": early_stopping_metric,
                "monitor_score": monitor_score,
                "val_computed": bool(should_validate),
                "lr_backbone": optimizer.param_groups[0]["lr"],
                "lr_head": optimizer.param_groups[1]["lr"],
                "elapsed_s": round(elapsed, 2),
            }
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_macro_f1": val_macro_f1,
                "val_gmean": val_gmean,
                "monitor_metric": early_stopping_metric,
                "monitor_score": monitor_score,
                "val_computed": bool(should_validate),
            }
        )

        if (
            should_validate
            and (epoch + 1) > early_stopping_warmup_epochs
            and epochs_no_improve >= CFG.ft_early_stopping_patience
        ):
            print(f"[{model_name}] Early stopping at epoch {epoch+1}")
            break

        # Epoch 缁撴潫鍚庢洿鏂?LR Scheduler
        if scheduler is not None:
            scheduler.step()

    if last_checkpoint_averaging and len(top_checkpoints) > 0:
        selected_meta = top_checkpoints[:checkpoint_top_k]
        selected_payloads: List[Dict[str, Any]] = []
        for item in selected_meta:
            path = Path(str(item["path"]))
            if not path.exists():
                continue
            selected_payloads.append(torch.load(path, map_location="cpu"))
        if len(selected_payloads) > 0:
            averaged_state = _average_state_dicts(
                [payload["state_dict"] for payload in selected_payloads]
            )
            best_payload = dict(selected_payloads[0])
            best_payload["state_dict"] = averaged_state
            best_payload["checkpoint_averaging_applied"] = bool(
                len(selected_payloads) > 1
            )
            best_payload["averaged_from"] = [
                str(Path(str(item["path"]))) for item in selected_meta
            ]
            torch.save(best_payload, ckpt_path)
            print(
                f"[{model_name}] checkpoint averaging applied | "
                f"num_checkpoints={len(selected_payloads)} saved={ckpt_path}"
            )

    if log_fp is not None:
        log_fp.close()
    
    if best_epoch >= 0:
        print(
            f"[{model_name}] 寰皟瀹屾垚 | best_epoch={best_epoch+1} "
            f"best_val_f1={best_val_f1:.4f} best_val_gmean={best_val_gmean:.4f} "
            f"best_monitor({early_stopping_metric})={best_monitor_score:.4f}"
        )
    else:
        print(f"[{model_name}] 寰皟瀹屾垚锛屼絾鏈幏寰楁湁鏁堥獙璇佺粨鏋滐紙best checkpoint 鏈洿鏂帮級")
    
    return FineTuneResult(
        model_name=model_name,
        best_epoch=best_epoch,
        best_val_macro_f1=float(best_val_f1),
        best_val_gmean=float(best_val_gmean),
        checkpoint_path=str(ckpt_path),
        history=history,
    )

