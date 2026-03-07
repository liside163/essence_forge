# 淬真 / Essence Forge

基于rfymad数据集进行11分类故障诊断研究。

`essence_forge` 现在已经整理成一个可独立运行的实验包。你可以把整个 `essence_forge/` 目录复制到另一台机器上，直接运行，而不再依赖仓库根目录下的 `uav_tl`、`tools` 或 `main.py`。

运行时依赖包括：`torch`、`numpy`、`pandas`、`scikit-learn`，以及用于绘制评估图表的 `matplotlib`。

## 目录结构

- `config.py`：内置实验配置加载器与运行时快照写入工具
- `split.py`：数据集索引与划分辅助函数
- `preprocess.py`：源域统计、阶段窗口、健康掩码、残差特征与本地预计算
- `features.py`：FFT 幅值与 token 池化辅助函数
- `model.py`：`EssenceForgeTCN`
- `run.py`：`split`、`preprocess`、`train`、`finetune`、`eval`、`pipeline`
- `core/`：训练、微调与评估使用的内置运行时模块
- `configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json`：内置且便于阅读的配置文件

## 可移植使用方式

1. 复制整个 `essence_forge/` 目录。
2. 编辑 `configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json`，把 `paths.data_dir` 改成目标机器上的数据集路径。
3. 进入复制后的 `essence_forge/` 目录，并在该目录下执行命令。

```bash
cd essence_forge

python run.py pipeline \
  --config configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json \
  --run-dir outputs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s
```

如果你需要重新生成缓存的预处理产物，可以执行：

```bash
cd essence_forge

python run.py pipeline \
  --config configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json \
  --run-dir outputs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s \
  --force-rebuild
```
