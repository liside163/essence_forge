"""
losses.py

损失函数集合

包含:
- LabelSmoothingCrossEntropy: Label Smoothing
- FocalLoss: 解决类别不平衡
- AdaptiveCostSensitiveLoss: ACS-ATCN 论文核心损失
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing 损失函数
    
    作用: 将 hard label (0,1) 平滑为 soft label (α/C, 1-α)
    优势: 减少模型对训练标签的过度自信，提高泛化能力
    
    公式:
        soft_targets = (1-α) * one_hot + α / num_classes
        loss = -sum(soft_targets * log_softmax(logits))
    
    维度变换:
        logits: [B, C]
        targets: [B]
        loss: 标量
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None
    ):
        """
        参数:
            smoothing: 平滑系数 α (0 表示无平滑)
            weight: 可选的类别权重 [C]
        """
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算损失
        
        参数:
            logits: [B, C] 模型输出
            targets: [B] 真实标签
        """
        num_classes = logits.size(-1)
        
        # 计算 log softmax
        log_probs = F.log_softmax(logits, dim=-1)  # [B, C]
        
        # 构建 soft targets
        # one_hot: [B, C]
        one_hot = torch.zeros_like(logits).scatter_(
            dim=-1,
            index=targets.unsqueeze(-1),
            value=1.0
        )
        
        # soft_targets = (1 - α) * one_hot + α / C
        soft_targets = (1.0 - self.smoothing) * one_hot + self.smoothing / num_classes
        
        # 计算交叉熵
        # loss = -sum(soft_targets * log_probs)
        loss = -(soft_targets * log_probs).sum(dim=-1)  # [B]
        
        # 应用类别权重
        if self.weight is not None:
            weight = self.weight.to(logits.device)
            loss = loss * weight[targets]
        
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss
    
    作用: 通过降低易分样本的权重，聚焦于难分样本
    论文: "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    公式:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    参数:
        alpha: 类别权重 [C] 或标量
        gamma: 聚焦参数，γ > 0 时减少易分样本权重
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        参数:
            logits: [B, C]
            targets: [B]
        """
        probs = F.softmax(logits, dim=-1)  # [B, C]
        
        # 获取目标类别的概率
        # p_t: [B]
        p_t = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # 计算 focal weight: (1 - p_t)^γ
        focal_weight = (1.0 - p_t) ** self.gamma  # [B]
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # [B]
        
        # 应用 focal weight
        loss = focal_weight * ce_loss  # [B]
        
        # 应用 alpha 权重
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            loss = alpha_t * loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class AdaptiveCostSensitiveLoss(nn.Module):
    """
    自适应类别成本敏感损失函数
    
    ACS-ATCN 论文核心创新：
    - 类别成本权重作为可优化变量
    - 通过 Optuna 搜索最优成本组合
    - 而非固定的 inverse_sqrt / inverse_freq
    
    公式:
        L = sum_i (cost_i * CE(logits, targets)) / sum(costs)
    
    参数:
        base_costs: 初始成本权重 [C]
        label_smoothing: 可选的 label smoothing
    """
    
    def __init__(
        self,
        num_classes: int,
        base_costs: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
        # 初始化成本（默认均匀）
        if base_costs is None:
            base_costs = torch.ones(num_classes)
        
        # 注册为 buffer（不作为参数训练，但随模型移动到 GPU）
        self.register_buffer("costs", base_costs)
    
    def set_costs(self, costs: torch.Tensor) -> None:
        """运行时更新类别成本（供 Optuna 调用）"""
        self.costs = costs.to(self.costs.device)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        参数:
            logits: [B, C]
            targets: [B]
        """
        # 获取每个样本的成本权重
        sample_costs = self.costs[targets]  # [B]
        
        # 计算基础交叉熵
        if self.label_smoothing > 0:
            log_probs = F.log_softmax(logits, dim=-1)
            one_hot = torch.zeros_like(logits).scatter_(
                dim=-1,
                index=targets.unsqueeze(-1),
                value=1.0
            )
            soft_targets = (1.0 - self.label_smoothing) * one_hot + \
                           self.label_smoothing / self.num_classes
            ce_loss = -(soft_targets * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction="none")
        
        # 加权损失
        weighted_loss = sample_costs * ce_loss

        return weighted_loss.mean()


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss

    给少数类(Minority Classes)提供更大的分类边缘(Margin)，
    强制模型在少数类上学到更具区分性的边界。

    论文: "Long-Tail Recognition via Weight Balancing" (ICLR 2022)

    公式:
        m_j = C / (n_j^{1/4})  (类别 margin)
        loss = CrossEntropy(s * (logits - margin), targets)

    参数:
        cls_num_list: 每个类别的样本数列表 [C]
        max_m: 最大 margin 值
        weight: 可选的类别权重 [C]
        s: 缩放因子（类似 ArcFace）

    维度变换:
        inputs: [Batch_Size, Num_Classes] - 模型输出的原始 Logits
        targets: [Batch_Size] - 真实的类别索引
        output: scalar loss
    """

    def __init__(
        self,
        cls_num_list: list,
        max_m: float = 0.5,
        weight: Optional[torch.Tensor] = None,
        s: float = 30.0,
    ):
        super().__init__()
        # 计算每个类别的 Margin: m_j = C / (n_j^{1/4})
        cls_num_list = np.array(cls_num_list, dtype=np.float64)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.s = s  # 缩放因子
        self.weight = weight

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算损失

        参数:
            x: [B, C] 模型输出 logits
            target: [B] 真实标签
        """
        # 前向时统一对齐到 logits 所在设备，避免 m_list/weight 与 x 设备不一致。
        target = target.to(device=x.device, dtype=torch.long)
        index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        index.scatter_(1, target.view(-1, 1), True)

        index_float = index.to(dtype=x.dtype)
        m_list = self.m_list.to(device=x.device, dtype=x.dtype)
        # 选出当前batch中每个样本对应类别的margin: [B, 1]
        batch_m = torch.matmul(m_list.unsqueeze(0), index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))

        # 将 Margin 应用到正确类别的 Logit 上
        x_m = x - batch_m

        # 组合原始 Logits 和带有 Margin 的 Logits
        output = torch.where(index, x_m, x)

        # 使用 CrossEntropy 计算带权重的损失
        weight = self.weight.to(x.device) if self.weight is not None else None
        return F.cross_entropy(self.s * output, target, weight=weight)


class ClassBalancedFocalLoss(nn.Module):
    """
    类平衡焦点损失：针对长尾分布的损失函数。

    论文: "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)

    公式:
        L = - (1-β)/(1-β^n_y) × (1-p_y)^γ × log(p_y)

    其中:
    - β: 超参数 (通常 0.9, 0.99, 0.999)
    - n_y: 类别 y 的样本数
    - γ: 焦点参数 (通常 2.0)

    功能:
    - 有效数量权重：解决类别不平衡
    - 焦点调制：关注难分类样本
    """

    def __init__(
        self,
        samples_per_class: list,
        num_classes: int = 11,
        beta: float = 0.999,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        参数:
            samples_per_class: 每个类的样本数列表
            num_classes: 类别数
            beta: 有效数量超参数
            gamma: 焦点参数
            reduction: 'mean' 或 'sum' 或 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if len(samples_per_class) != num_classes:
            raise ValueError(
                f"samples_per_class 长度 ({len(samples_per_class)}) "
                f"必须等于 num_classes ({num_classes})"
            )

        # 计算有效数量权重
        samples = np.array(samples_per_class, dtype=np.float64)
        effective_num = 1.0 - np.power(beta, samples)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / weights.sum() * num_classes  # 归一化

        self.register_buffer(
            "weights",
            torch.tensor(weights, dtype=torch.float32)
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            logits: [B, C] 模型输出
            targets: [B] 真实标签

        返回:
            loss: 标量 (或 [B] 如果 reduction='none')
        """
        # 计算交叉熵 (不 reduction)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # 计算焦点调制因子
        pt = torch.exp(-ce_loss)  # 正确类别的概率
        focal_factor = (1.0 - pt) ** self.gamma

        # 应用类权重
        weights = self.weights.to(logits.device)
        class_weights = weights[targets]

        # 最终损失
        loss = focal_factor * ce_loss * class_weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ConfusionAwareLoss(nn.Module):
    """
    混淆感知损失

    在基础损失上叠加"定向混淆惩罚"，重点抑制已知高频混淆对：
    (真实类别 src -> 易混淆类别 dst)。

    公式:
        L = L_base + λ * mean(penalty_i)
        penalty_i = sum_k scale_k * p(dst_k | x_i), 当 y_i == src_k

    参数:
        base_loss: 基础损失函数（可传 FocalLoss / CrossEntropy 等）
        confusion_pairs: 混淆对列表，元素为 (src, dst, scale)
        lambda_confusion: 混淆惩罚整体权重 λ
    """

    DEFAULT_CONFUSION_PAIRS: list[tuple[int, int, float]] = [
        (8, 10, 3.0),  # barometer -> low_voltage
        (8, 9, 2.5),   # barometer -> GPS
        (10, 8, 2.5),  # low_voltage -> barometer
        (9, 0, 2.0),   # GPS -> motor
        (0, 9, 2.0),   # motor -> GPS
    ]

    def __init__(
        self,
        base_loss: Optional[nn.Module] = None,
        confusion_pairs: Optional[list[tuple[int, int, float]]] = None,
        lambda_confusion: float = 1.0,
    ):
        super().__init__()
        self.base_loss = (
            base_loss
            if base_loss is not None
            else nn.CrossEntropyLoss(reduction="none")
        )
        self.lambda_confusion = lambda_confusion

        pairs = (
            confusion_pairs
            if confusion_pairs is not None
            else self.DEFAULT_CONFUSION_PAIRS
        )
        if len(pairs) == 0:
            raise ValueError("confusion_pairs 不能为空")

        # 注册为 buffer，保证随模型移动到同一 device
        self.register_buffer(
            "src_classes",
            torch.tensor([p[0] for p in pairs], dtype=torch.long),
        )
        self.register_buffer(
            "dst_classes",
            torch.tensor([p[1] for p in pairs], dtype=torch.long),
        )
        self.register_buffer(
            "penalty_scales",
            torch.tensor([p[2] for p in pairs], dtype=torch.float32),
        )

    def _reduce_base_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """将基础损失统一为标量。"""
        base = self.base_loss(logits, targets)
        if base.ndim == 0:
            return base
        return base.mean()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            logits: [B, C] 模型输出
            targets: [B] 真实标签

        返回:
            标量损失
        """
        targets = targets.to(device=logits.device, dtype=torch.long)
        base_loss = self._reduce_base_loss(logits, targets)

        # 计算每个样本在"易混淆目标类别"上的概率
        probs = F.softmax(logits, dim=-1)  # [B, C]
        confusion_penalty = logits.new_zeros(targets.size(0))

        for src_cls, dst_cls, scale in zip(
            self.src_classes,
            self.dst_classes,
            self.penalty_scales,
        ):
            sample_mask = targets == src_cls
            if sample_mask.any():
                # 仅对真实类别命中该混淆对的样本施加惩罚
                confusion_penalty[sample_mask] += (
                    scale.to(dtype=logits.dtype)
                    * probs[sample_mask, dst_cls]
                )

        # 若当前 batch 不包含任何配置中的 src 类别，则该项自然为 0
        confusion_loss = confusion_penalty.mean()
        return base_loss + self.lambda_confusion * confusion_loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss（监督式对比损失）

    目标:
    - 拉近同类样本在特征空间中的距离
    - 推远不同类样本在特征空间中的距离
    - 可与分类损失直接相加组合使用

    参考公式:
        L_i = -1/|P(i)| * Σ_{p∈P(i)} log(
            exp(sim(z_i, z_p)/τ) / Σ_{a≠i} exp(sim(z_i, z_a)/τ)
        )

    参数:
        temperature: 温度系数 τ
        proj_head: 可选投影头维度，例如 [64, 128, 128]；为 None 时不使用投影头
        normalize: 是否对投影后的特征做 L2 归一化
        eps: 数值稳定项，避免 log(0)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        proj_head: Optional[list[int]] = None,
        normalize: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")

        self.temperature = temperature
        self.normalize = normalize
        self.eps = eps
        self.proj_head = self._build_projection_head(proj_head)

    @staticmethod
    def _build_projection_head(
        proj_dims: Optional[list[int]],
    ) -> Optional[nn.Sequential]:
        """
        构建可选投影头（MLP）。

        例如:
            proj_dims=[64, 128, 128]
            对应 Linear(64->128) + ReLU + Linear(128->128)
        """
        if proj_dims is None:
            return None
        if len(proj_dims) < 2:
            raise ValueError("proj_head 至少需要 2 个维度，例如 [64, 128]")

        layers: list[nn.Module] = []
        for i in range(len(proj_dims) - 1):
            in_dim = proj_dims[i]
            out_dim = proj_dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(proj_dims) - 2:
                layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            features: [B, D] 输入特征
            targets: [B] 类别标签

        返回:
            标量损失（可与分类损失组合：total = ce + λ * supcon）
        """
        if features.ndim != 2:
            raise ValueError(f"features 维度应为 [B, D]，当前为 {features.shape}")
        if targets.ndim != 1:
            raise ValueError(f"targets 维度应为 [B]，当前为 {targets.shape}")
        if features.size(0) != targets.size(0):
            raise ValueError("features 和 targets 的 batch 大小不一致")

        device = features.device
        dtype = features.dtype
        batch_size = features.size(0)
        targets = targets.to(device=device, dtype=torch.long)

        # 可选投影头
        if self.proj_head is not None:
            features = self.proj_head(features)

        # 对比学习中常见做法：单位球归一化后使用点积即余弦相似度
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)

        # 相似度矩阵: [B, B]
        logits = torch.matmul(features, features.T) / self.temperature

        # 数值稳定：每行减去最大值，不影响 softmax 结果
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # 同类样本掩码（包含对角线）
        positive_mask = (targets.unsqueeze(0) == targets.unsqueeze(1)).to(dtype=dtype)

        # 去掉自身对比项 i==i
        self_mask = torch.eye(batch_size, device=device, dtype=dtype)
        positive_mask = positive_mask * (1.0 - self_mask)
        logits_mask = 1.0 - self_mask

        # log_prob(i, a) = logits(i, a) - log(sum_{a!=i} exp(logits(i, a)))
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps))

        # 对每个 anchor，仅在正样本集合上求均值
        positive_count = positive_mask.sum(dim=1)  # [B]
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1.0)

        # 只对存在正样本的 anchor 计算损失，避免单样本类别造成 NaN
        valid_anchors = positive_count > 0
        if valid_anchors.any():
            loss = -mean_log_prob_pos[valid_anchors].mean()
        else:
            # 若整个 batch 没有任何正对，返回 0 以保证训练流程可继续
            loss = torch.zeros((), device=device, dtype=dtype)

        return loss


class CenterLoss(nn.Module):
    """
    Center Loss：约束样本向所属类别中心收缩。

    公式：
        L = 0.5 * mean(||x_i - c_{y_i}||^2)

    支持：
    - 仅对 active_class_ids 内类别生效（用于“混淆对定向”场景）
    - 与主分类损失加权求和：total = ce + λ * center_loss
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        active_class_ids: Optional[list[int]] = None,
    ):
        super().__init__()
        if int(num_classes) <= 0:
            raise ValueError("num_classes 必须 > 0")
        if int(feat_dim) <= 0:
            raise ValueError("feat_dim 必须 > 0")

        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

        active = sorted({int(v) for v in (active_class_ids or [])})
        if len(active) == 0:
            active_tensor = torch.empty((0,), dtype=torch.long)
        else:
            for class_id in active:
                if class_id < 0 or class_id >= self.num_classes:
                    raise ValueError(
                        f"active_class_ids 含非法类 ID={class_id}，合法范围=[0,{self.num_classes - 1}]"
                    )
            active_tensor = torch.tensor(active, dtype=torch.long)
        self.register_buffer("active_class_ids", active_tensor)

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features 维度应为 [B, D]，当前={tuple(features.shape)}")
        if int(features.size(1)) != self.feat_dim:
            raise ValueError(
                f"features 的 feat_dim 不匹配，期望={self.feat_dim}，当前={int(features.size(1))}"
            )
        if targets.ndim != 1:
            raise ValueError(f"targets 维度应为 [B]，当前={tuple(targets.shape)}")
        if int(features.size(0)) != int(targets.size(0)):
            raise ValueError("features 和 targets 的 batch 大小不一致")

        targets = targets.to(device=features.device, dtype=torch.long)
        if targets.numel() == 0:
            return features.new_zeros(())
        if int(targets.min().item()) < 0 or int(targets.max().item()) >= self.num_classes:
            raise ValueError(
                f"targets 含非法类 ID，合法范围=[0,{self.num_classes - 1}]"
            )

        filtered_features = features
        filtered_targets = targets
        if self.active_class_ids.numel() > 0:
            active = self.active_class_ids.to(device=features.device)
            mask = (targets.unsqueeze(1) == active.unsqueeze(0)).any(dim=1)
            if not bool(mask.any()):
                return features.new_zeros(())
            filtered_features = features[mask]
            filtered_targets = targets[mask]

        centers_batch = self.centers.index_select(0, filtered_targets)
        distances = (filtered_features - centers_batch).pow(2).sum(dim=1)
        return 0.5 * distances.mean()
