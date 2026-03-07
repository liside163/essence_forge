"""
precomputed_dataset.py

妫板嫯缁犳鐗遍張鏆熼幑娉?閳ユ柡鈧?缁傝崵鍤庣紓鎾崇摠閸旂娀鈧喕缂佸啰缁捐￥鈧?

鐠佹崘鐟曚胶鍋ｉ敍?
1. 鐠佺矊閸撳秳绔村▎鈩冣偓褍鐨㈢涵鐣鹃幀褍閻炲棴绱欓崝鐘烘祰閵嗕礁鍨忕粣妞尖偓浣哥秺娑撯偓閸栨牓鈧礁浠存惔閿嬪负閻焦瀚鹃幒銉礆閸愭瑥鍙?.npy閿?
2. 鐠佺矊閺?`__getitem__` 娴犲懎浠?np.load + 閸欌偓澶婃躬缁惧灝瀵尨绱濇径褍绠欓崙蹇撶毌 DataLoader CPU 瀵偓闁库偓閿?
3. 婢х偛宸遍幙宥勭稊閿涘牓娈㈤張鐑樷偓褔鍎撮崚鍡礆娣囨繃瀵旈崷銊у殠閿涘奔绻氱拠浣圭槨 epoch 娴溠呮晸娑撳秴鎮撻幍鏉垮З閵?
"""

from __future__ import annotations

import json
import errno
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import sys


from essence_forge.core.runtime_config import CFG
from essence_forge.core.channel_layout import LEGACY_CORE_CHANNEL_COUNT, channel_names_from_specs
from essence_forge.core.datasets import Sample, collate_padded
from essence_forge.core.augment import (
    AugContext,
    PhysicsAwareAugmentor,
    HardClassMixupAugmentor,
    HardClassMixupConfig,
    build_augmentor_from_config,
)
from essence_forge.core.utils import stable_seed_from_items


@dataclass(frozen=True)
class PrecomputedSampleEntry:
    """manifest 娑撳礋閺夆剝鐗遍張娈戦崗鍐т繆閹偓"""

    path: str
    class_id: int
    length: int
    window_length: int
    # 娴犮儰绗呯€涙閻劋绨崷銊у殠婢х偛宸遍弮鍫曞櫢瀵?AugContext
    fault_code: int
    fault_onset_idx: int
    window_start: int


def load_manifest(manifest_path: Path) -> Dict:
    """鐠囪褰?manifest.json 楠炶埖鐗庢宀€澧楅張鈧"""

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("version", 1) != 1:
        raise ValueError(f"娑撳秵鏁幐浣烘畱 manifest 閻楀牊婀? {manifest.get('version')}")
    return manifest


def parse_manifest_entries(manifest: Dict, base_dir: Path) -> List[PrecomputedSampleEntry]:
    """Parse manifest items and resolve file paths under the local split dir.

    Manifest path strings may come from another OS (for example Windows paths
    loaded on Linux/WSL). We normalize separators and always map to
    `base_dir/<file_name>`.
    """

    entries: List[PrecomputedSampleEntry] = []
    for item in manifest["samples"]:
        raw_path = str(item["path"]).strip().strip("\"")
        file_name = Path(raw_path.replace("\\", "/")).name
        local_path = base_dir / file_name

        entries.append(
            PrecomputedSampleEntry(
                path=str(local_path),
                class_id=int(item["class_id"]),
                length=int(item["length"]),
                window_length=int(item["window_length"]),
                fault_code=int(item.get("fault_code", 10)),
                fault_onset_idx=int(item.get("fault_onset_idx", -1)),
                window_start=int(item.get("window_start", 0)),
            )
        )
    return entries


class PrecomputedDataset(Dataset):
    """
    妫板嫯缁犳鐗遍張鏆熼幑娉﹂妴?

    __getitem__ 鐠虹窞:
    1. np.load(sample.npy) -> x: [T, C]  (瀹告彃缍婃稉鈧崠?+ 閸嬨儱鎮嶉幒鈺冪垳閹峰吋甯?
    2. (閸欌偓? 閸︺劎鍤庢晶鐐插繁: 娴犲懎閸?raw_channels 闁岸浜鹃弬钘夊
    3. 鏉╂柨娲?Sample(x, length, y)

    娑?SourceDomainDataset 閸忕厧閻ㄥ嫭甯撮崣?
    - set_epoch(epoch)
    - set_augmentor_policy(policy)
    - update_class_recall(class_recall)
    - stage_records (鐏炵偞鈧? 閻劋绨?build_sampler_for_train_dataset)
    """

    def __init__(
        self,
        precomputed_dir: Path,
        is_train: bool = True,
        base_seed: int = 42,
        enable_physics_augment: bool = True,
        augment_mode: str = "full",
        use_mmap: bool = False,
    ):
        self.precomputed_dir = Path(precomputed_dir).resolve()
        manifest_path = self.precomputed_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json 娑撳秴鐡ㄩ崷? {manifest_path}")

        manifest = load_manifest(manifest_path)
        self.entries = parse_manifest_entries(manifest, self.precomputed_dir)
        self.raw_channels = int(manifest.get("raw_channels", 19))
        self.channel_names = tuple(manifest.get("channel_names", []))

        self.is_train = bool(is_train)
        self.base_seed = int(base_seed)
        self._epoch = 0
        self.use_mmap = bool(use_mmap)
        self.io_max_retries = max(1, int(getattr(CFG, "precomputed_io_max_retries", 3)))
        self.io_retry_delay_sec = max(
            0.0,
            float(getattr(CFG, "precomputed_io_retry_delay_sec", 0.02)),
        )

        # 婢х偛宸遍柊宥囩枂
        self.augment_mode = str(augment_mode).strip().lower()
        if not self.is_train:
            self.augment_mode = "off"
        if self.augment_mode not in {"full", "light", "off"}:
            raise ValueError("augment_mode 韫囧懘銆忛弰?full/light/off")

        self.enable_physics_augment = bool(enable_physics_augment and self.is_train)
        self.physics_augmentor: Optional[PhysicsAwareAugmentor] = None
        self.hard_class_mixup: Optional[HardClassMixupAugmentor] = None

        if self.enable_physics_augment:
            self._init_physics_augmentor()

        # 閸忕厧 train.py 娑?build_sampler_for_train_dataset 閻ㄥ嫭甯撮崣?
        self._build_compat_stage_records()

    def _build_compat_stage_records(self) -> None:
        """閺嬪嫬缂撻崗鐓?train.py sampler 閹碘偓闂団偓閻?stage_records 鐏炵偞鈧佲偓"""

        from essence_forge.core.datasets import StageWindowRecord

        self.stage_records = [
            StageWindowRecord(
                record_idx=0,
                start=entry.window_start,
                window_length=entry.window_length,
                class_id=entry.class_id,
            )
            for entry in self.entries
        ]

    def _init_physics_augmentor(self) -> None:
        """閸掓繂閸栨牜澧块悶鍡樺妳閻儱瀵搫娅掗敍鍫滅瑢 SourceDomainDataset 闁槒绶稉鈧懛杈剧礆閵"""

        if not self.is_train or self.augment_mode in {"off", "light"}:
            return

        strategy = str(getattr(CFG, "augment_strategy", "legacy")).lower()
        if strategy != "physics_aware":
            return

        self.physics_augmentor = build_augmentor_from_config(CFG)

        if bool(
            getattr(
                CFG,
                "augment_class_mixup_enable",
                getattr(CFG, "augment_hard_class_mixup_enable", False),
            )
        ):
            self.hard_class_mixup = HardClassMixupAugmentor(
                cfg=HardClassMixupConfig(
                    enable=True,
                    prob=float(
                        getattr(
                            CFG,
                            "augment_class_mixup_prob",
                            getattr(CFG, "augment_hard_class_mixup_prob", 0.35),
                        )
                    ),
                    alpha=float(
                        getattr(
                            CFG,
                            "augment_class_mixup_alpha",
                            getattr(CFG, "augment_hard_class_mixup_alpha", 0.3),
                        )
                    ),
                    target_classes=tuple(
                        int(v)
                        for v in getattr(
                            CFG,
                            "augment_class_mixup_target_classes",
                            getattr(CFG, "augment_hard_class_targets", (5, 6, 7)),
                        )
                    ),
                    same_fault_family_only=bool(
                        getattr(CFG, "augment_class_mixup_same_fault_family_only", False)
                    ),
                    max_bank_size_per_class=int(
                        getattr(CFG, "augment_class_mixup_max_bank_size_per_class", 128)
                    ),
                )
            )

    def set_epoch(self, epoch: int) -> None:
        """鐠佸墽鐤嗚ぐ鎾冲 epoch閿涘牆鍚嬬€?SourceDomainDataset 閹恒儱褰涢敍澶堚偓"""
        if self.hard_class_mixup is not None and int(epoch) != self._epoch:
            self.hard_class_mixup.clear_bank()
        self._epoch = int(epoch)

    def set_augmentor_policy(self, policy) -> None:
        """鐠佸墽鐤嗘晶鐐插繁缁涙牜鏆愰敍鍫濆悑鐎圭懓娲栭柅鈧張鍝勫煑閹恒儱褰涢敍澶堚偓"""
        if self.physics_augmentor is not None:
            self.physics_augmentor.set_policy(policy)

    def update_class_recall(self, class_recall: Dict[int, float]) -> None:
        """閺囧瓨鏌婇崥鍕閸欐礀閻滃浄绱欓崗鐓庨崝銊︹偓浣稿鐑樺复閸欙綇绱氶妴"""
        if self.physics_augmentor is not None:
            self.physics_augmentor.update_class_recall(class_recall)

    def __len__(self) -> int:
        return len(self.entries)

    def _load_precomputed_array(self, sample_path: str, index: int) -> np.ndarray:
        """
        Robust .npy loader for worker mode.

        On WSL/DrvFs, transient OSError(EINVAL/EIO/EMFILE/ENFILE/ENOMEM) can
        appear under multi-process DataLoader pressure. We retry with small
        backoff and raise a detailed error if it keeps failing.
        """
        retryable_errnos = {
            errno.EINVAL,
            errno.EIO,
            errno.ENOMEM,
            errno.EMFILE,
            errno.ENFILE,
        }
        last_exc: Optional[BaseException] = None

        for attempt in range(1, self.io_max_retries + 1):
            try:
                if self.use_mmap:
                    return np.load(sample_path, mmap_mode="r", allow_pickle=False).astype(np.float32)

                with open(sample_path, "rb") as f:
                    return np.load(f, allow_pickle=False).astype(np.float32)
            except OSError as exc:
                last_exc = exc

                # Fallback: let NumPy handle opening directly.
                if not self.use_mmap:
                    try:
                        return np.load(sample_path, allow_pickle=False).astype(np.float32)
                    except Exception as fallback_exc:
                        last_exc = fallback_exc

                if exc.errno not in retryable_errnos or attempt >= self.io_max_retries:
                    break
            except Exception as exc:
                last_exc = exc
                if attempt >= self.io_max_retries:
                    break

            time.sleep(self.io_retry_delay_sec * attempt)

        file_info = "unknown"
        try:
            p = Path(sample_path)
            if p.exists():
                file_info = f"exists,size={p.stat().st_size}"
            else:
                file_info = "missing"
        except Exception:
            pass

        raise OSError(
            f"Failed to load precomputed sample after {self.io_max_retries} attempt(s): "
            f"index={index}, path='{sample_path}', file_info={file_info}, last_error={last_exc!r}"
        ) from last_exc

    def _rng_for_index(self, index: int) -> np.random.Generator:
        """娑撶儤鐗遍張鍌ㄥ鏇熺€柅鐘插讲婢跺秶骞囬梾蹇旀簚閺佹壆鏁撻幋鎰珤閵"""
        seed = stable_seed_from_items(self.base_seed, self._epoch, int(index))
        return np.random.default_rng(seed)

    def _apply_augment(
        self,
        x: np.ndarray,
        y: int,
        rng: np.random.Generator,
        epoch: int = 0,
        context: Optional[AugContext] = None,
    ) -> np.ndarray:
        """
        閸︺劎鍤庢晶鐐插繁閿涘牅绮庢担婊呮暏娴滃骸澧?raw_channels 闁岸浜鹃敍澶堚偓?
        闁槒绶稉?SourceDomainDataset._apply_augment 娣囨繃瀵旀稉鈧懛娣偓?
        """
        if self.augment_mode == "off":
            return x

        if self.augment_mode == "light":
            sigma = max(float(CFG.augment_gaussian_sigma) * 0.5, 0.0)
            if sigma > 0:
                noise = rng.normal(0.0, sigma, x.shape)
                x = x + noise.astype(np.float32)

            if CFG.enable_drift_noise and CFG.augment_drift_sigma > 0:
                T, C = x.shape
                slopes = rng.normal(0.0, float(CFG.augment_drift_sigma) * 0.25, (1, C))
                t_normalized = np.linspace(0.0, 1.0, T, dtype=np.float32).reshape(-1, 1)
                drift = (slopes * t_normalized).astype(np.float32)
                x = x + drift
            return x

        if self.physics_augmentor is not None:
            return self.physics_augmentor.augment(x, y, epoch, rng, context=context)

        # Fallback: 娴肩姷绮虹粻鈧崡鏇炲?
        if CFG.augment_gaussian_sigma > 0:
            noise = rng.normal(0.0, float(CFG.augment_gaussian_sigma), x.shape)
            x = x + noise.astype(np.float32)

        if CFG.enable_drift_noise and CFG.augment_drift_sigma > 0:
            T, C = x.shape
            slopes = rng.normal(0.0, float(CFG.augment_drift_sigma), (1, C))
            t_normalized = np.linspace(0.0, 1.0, T, dtype=np.float32).reshape(-1, 1)
            drift = (slopes * t_normalized).astype(np.float32)
            x = x + drift

        return x

    def __getitem__(self, index: int) -> Sample:
        entry = self.entries[index]

        x = self._load_precomputed_array(entry.path, index=index)

        # 閸︺劎鍤庢晶鐐插繁閿涘牅绮庣拋绮岄張鐕傜礉娴犲懎閸?raw_channels 闁岸浜鹃敍?
        if self.is_train and self.augment_mode != "off" and bool(getattr(CFG, "enable_augment", True)):
            rng = self._rng_for_index(index)
            raw_part = x[:, : self.raw_channels]
            legacy_core_channels = min(LEGACY_CORE_CHANNEL_COUNT, int(raw_part.shape[1]))
            raw_core = raw_part[:, :legacy_core_channels].copy()
            raw_tail = raw_part[:, legacy_core_channels:].copy()

            aug_context = AugContext(
                fault_code=entry.fault_code,
                fault_onset_idx=entry.fault_onset_idx,
                window_start=entry.window_start,
                window_length=entry.window_length,
                channel_names=(
                    tuple(self.channel_names[:legacy_core_channels])
                    if self.channel_names
                    else channel_names_from_specs(getattr(CFG, "channels", ()))[:legacy_core_channels]
                ),
            )
            raw_core = self._apply_augment(
                raw_core,
                y=entry.class_id,
                rng=rng,
                epoch=self._epoch,
                context=aug_context,
            )

            # P3: 绾 Mixup
            if self.hard_class_mixup is not None:
                self.hard_class_mixup.register_sample(y=entry.class_id, x=raw_core)
                raw_core = self.hard_class_mixup.maybe_mixup(
                    x=raw_core,
                    y=entry.class_id,
                    rng=rng,
                )

            x[:, : self.raw_channels] = np.concatenate(
                [raw_core, raw_tail],
                axis=1,
            ).astype(np.float32, copy=False)

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.tensor(entry.class_id, dtype=torch.long)
        return Sample(x=x_tensor, length=entry.length, y=y_tensor)


