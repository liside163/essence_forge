# 基于rfymad数据集进行11分类故障诊断研究

`essence_forge` 是一个面向无人机遥测故障识别与跨域迁移学习的独立实验包。仓库已经整理成不依赖外部 `uav_tl`、`tools` 或 `main.py` 的独立代码包，核心入口是 `run.py`，默认配置示例是 `A2B0 -> A2B1` 的源域训练 + 目标域微调实验。

需要注意：当前 GitHub 仓库只包含代码、配置和文档，不包含原始数据集。因此，别人把仓库克隆到本地后，可以查看代码、创建环境、阅读配置，但不能直接完成 `split`、`preprocess`、`train`、`finetune` 和 `eval` 的完整流程，除非另外拿到同结构的原始数据。

## 这个仓库在做什么

给定 RflyMAD 一类的无人机多传感器遥测数据，项目目标是：

- 从高噪声时序信号中提取对故障诊断有效的表征；
- 先在源域训练模型，再迁移到目标域微调；
- 输出准确率、Macro-F1、G-Mean、混淆矩阵和迁移差距分析结果。

当前内置示例配置对应的是一个明确的跨工况迁移任务：

- 只使用官方数据命名体系里的 `A=2`，也就是 `HIL` 子集；
- 源域使用 `B=0`，代码注释里对应 `hover` 工况；
- 目标域使用 `B=1`，代码注释里对应 `waypoint` 工况；
- 故障类别使用 `0..10` 全部类别，其中 `10` 被映射为 `no_fault`。

默认流水线是：

1. `split`：按照配置中的源域 / 目标域条件建立索引并切分训练、验证、测试集；
2. `preprocess`：计算源域归一化统计量，执行阶段窗口切分，并预计算训练/验证样本；
3. `train`：在源域训练基础模型；
4. `finetune`：加载源域 checkpoint，在目标域继续微调；
5. `eval`：导出评估指标、混淆矩阵和迁移效果汇总；
6. `pipeline`：串行执行以上全部步骤。

## 采用的方法

### 0. 数据前置处理与官方工具

本仓库不负责把原始官方日志转换成训练可用 CSV，也不在仓库内部执行升采样。按照当前实验配置和代码假设，数据准备应该分成两步：

1. 先使用官方数据处理工具把原始 HIL 数据整理成 `Case_*.csv` 格式；
2. 在官方工具侧把数据处理到本实验使用的时间分辨率，再把处理后的 CSV 目录作为 `paths.data_dir` 输入给本仓库。

可参考的官方资源：

- RflyMAD 数据集主页：<https://rfly-openha.github.io/documents/4_resources/dataset.html>
- 官方数据处理工具：<https://github.com/lerlis/Data_processing_tools>

根据官方工具 README：

- `--sub_dataset 2` 表示只处理 `HIL`
- `--flight_status 1` 表示 `hover`
- `--flight_status 2` 表示 `waypoint`
- `--fault_type 0` 表示处理全部故障类型
- `--trans_freq` 用来指定处理后文件频率，官方工具默认是 `20Hz`

如果你的目标是生成本仓库默认实验所需的 HIL `hover + waypoint` 两种工况，并把处理后频率设为 `120Hz`，README 可以按下面的方式说明：

```bash
git clone https://github.com/lerlis/Data_processing_tools.git
cd Data_processing_tools

python Rflytool_main.py \
  --original_path <RflyMAD原始数据目录> \
  --restore_path <处理后CSV输出目录> \
  --sub_dataset 2 \
  --flight_status 1 2 \
  --fault_type 0 \
  --trans_freq 120
```

说明：

- 上述命令会把 HIL 的 `hover` 和 `waypoint` 两种工况一起处理出来
- 本仓库再从这些处理后 CSV 中取源域和目标域
- 官方工具中的 topic 选择由其自带的 JSON 配置控制，例如 `data_HIL_GTD.json` 和 `data_HIL_PX4.json`

当前实验约定和配置是按 `120 Hz` 使用数据：

- 配置文件里显式写了 `windows.sample_rate_hz = 120`；
- 阶段窗口长度、窗口步长和部分派生特征都按 `120 Hz` 换算；
- 因此 README 应理解为：本仓库假定你提供的官方处理后数据已经能够按 `120 Hz` 方案使用，而不是由本仓库再做一次升采样。

作者本机默认数据目录是：

- `paths.data_dir = D:/Bigshe/RflyMAD_Dataset/process-spec-all`

这说明当前实验直接读取的是“已经处理好的 CSV 目录”，不是原始未整理日志。

### 1. 数据与样本构造

- 使用配置文件中的 `source_domain` 和 `target_domain` 定义跨域实验；
- 通过文件名 `Case_[A][B][CD][EFGHIJ].csv` 解析数据子集、工况和故障类型；
- 默认实验只保留 `A=2` 的 HIL 数据，并构造 `B=0 -> B=1` 的跨工况迁移；
- 对每个域内的数据按类别分层切分 `train / val / test`；
- 使用阶段窗口(`stage window`)从长时序中截取固定长度样本；
- 对源域训练集计算 z-score 统计量，并统一用于源域训练和目标域微调；
- 在基础传感器输入之外，额外构造健康掩码和跨传感器残差特征。

文件名解析规则在本仓库代码中是固定的：

- `A`：子数据集，注释中标明 `1=SIL, 2=HIL, 3=Real`
- `B`：飞行工况，当前代码注释中给出的示例是 `0=hover, 1=waypoint`
- `CD`：故障代码
- `EFGHIJ`：序列号

这里要区分两层编码：

- 官方工具命令行参数里，`flight_status` 用 `1/2` 选择 `hover/waypoint`
- 本仓库读取处理后文件时，依据 `Case_*.csv` 文件名中的 `B` 位做域过滤，当前代码注释将 `B=0/1` 解释为 `hover/waypoint`

默认划分配置来自 `configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json`：

- `train_ratio = 0.7`
- `val_ratio = 0.15`
- `test_ratio = 0.15`
- `split_seed = 42`
- `stratify_by_fault = true`

也就是说，源域和目标域会先分别过滤，再在各自域内按故障类别分层切成 `70% / 15% / 15%`。

默认配置里，窗口长度为 `60 / 120 / 240`，采样率为 `120 Hz`，阶段窗口时长为 `3.0 s`，步长为 `1.0 s`。

当前配置中的基础输入通道有 `30` 个，来自官方处理后 CSV 中的多种传感器字段，主要包括：

1. `accel_x` -> `_sensor_accel_0_x`
2. `accel_y` -> `_sensor_accel_0_y`
3. `accel_z` -> `_sensor_accel_0_z`
4. `gyro_x` -> `_sensor_gyro_0_x`
5. `gyro_y` -> `_sensor_gyro_0_y`
6. `gyro_z` -> `_sensor_gyro_0_z`
7. `mag_x` -> `_sensor_mag_0_x`
8. `mag_y` -> `_sensor_mag_0_y`
9. `mag_z` -> `_sensor_mag_0_z`
10. `pos_x` -> `_vehicle_local_position_0_x`
11. `pos_y` -> `_vehicle_local_position_0_y`
12. `pos_z` -> `_vehicle_local_position_0_z`
13. `vel_x` -> `_vehicle_local_position_0_vx`
14. `vel_y` -> `_vehicle_local_position_0_vy`
15. `vel_z` -> `_vehicle_local_position_0_vz`
16. `q0` -> `_vehicle_attitude_0_q[0]` 或 `_vehicle_attitude_0_q_0`
17. `q1` -> `_vehicle_attitude_0_q[1]` 或 `_vehicle_attitude_0_q_1`
18. `q2` -> `_vehicle_attitude_0_q[2]` 或 `_vehicle_attitude_0_q_2`
19. `q3` -> `_vehicle_attitude_0_q[3]` 或 `_vehicle_attitude_0_q_3`
20. `actuator_ctrl_0` -> `_actuator_controls_0_0_control[0]`
21. `actuator_ctrl_1` -> `_actuator_controls_0_0_control[1]`
22. `actuator_ctrl_2` -> `_actuator_controls_0_0_control[2]`
23. `actuator_ctrl_3` -> `_actuator_controls_0_0_control[3]`
24. `motor_rpm_1` -> `TrueState_data_motorRPMs[1]`
25. `motor_rpm_2` -> `TrueState_data_motorRPMs[2]`
26. `motor_rpm_3` -> `TrueState_data_motorRPMs[3]`
27. `motor_rpm_4` -> `TrueState_data_motorRPMs[4]`
28. `baro_alt` -> `_vehicle_air_data_0_baro_alt_meter`
29. `baro_temp` -> `_vehicle_air_data_0_baro_temp_celcius`
30. `baro_pressure` -> `_vehicle_air_data_0_baro_pressure_pa`

在此基础上，代码还会追加 `9` 个跨传感器残差通道，用于显式建模：

- `d(vel)/dt` 与 `accel` 的一致性
- `d(pos)/dt` 与 `vel` 的一致性
- `d²(pos)/dt²` 与 `accel` 的一致性

### 2. 模型结构

默认模型导出类是 `EssenceForgeTCN`，主体实现位于 `core/models/simplified_fft_lwpt_se_tcn.py`。结构可以概括为：

- 时域分支：`LWPT frontend + TCN`
- 频域分支：`FFT magnitude + Conv1d`
- 注意力机制：时域与频域分支分别使用 SE 通道注意力
- 融合方式：两支路池化后做 late fusion，再送入分类头

这套结构的目的，是同时保留：

- 时域动态模式；
- 频域能量分布；
- 多传感器之间的健康状态与残差信息。

### 3. 训练与迁移策略

- 源域训练阶段支持加权采样、类别不平衡损失和数据增强；
- 目标域阶段从源域 checkpoint 启动微调；
- 微调阶段支持冻结前端/部分层；
- 评估阶段支持阈值搜索，用于控制某些类别，尤其是 `no_fault` 的误判倾向。

### 4. 预处理细节

这个项目的“预处理”不是简单标准化，而是包含一整条和故障阶段对齐的数据准备流程：

1. `split`
   - 递归扫描 `paths.data_dir` 下的 `Case_*.csv`
   - 解析文件名中的 `A / B / fault_code`
   - 先按源域和目标域条件过滤，再按类别分层切分

2. `preprocess`
   - 使用源域训练集计算 z-score 统计量 `source_stats.json`
   - 生成阶段窗口索引
   - 预计算训练和验证阶段需要的窗口样本，写入 `precomputed/`

3. 阶段窗口标注策略
   - 若 `fault_code == 10`，整条任务都标为 `no_fault`
   - 若 `fault_code in 0..9`，故障注入前窗口标为 `no_fault`
   - 若 `fault_code in 0..9`，故障注入后窗口标为对应故障类
   - 当前配置打开了 `drop_prefault_normal_windows_for_fault_missions = true`，因此故障任务中的注入前正常窗口会被丢弃

4. 健康掩码
   - 对加速度计、陀螺仪、磁力计、气压计故障，在故障注入后把对应通道的健康状态置为 0
   - 该掩码会作为额外输入拼接到模型

5. 离线缓存
   - `MissionLoader` 会优先读取 `cache_dir` 下的 `.npy/.npz`
   - 缓存不存在或不匹配时，才回退到 CSV 解析
   - 这一步是加速机制，不是数据来源本身

## 仓库结构

- `run.py`：统一命令行入口，支持 `split/preprocess/train/finetune/eval/pipeline`
- `config.py`：实验配置加载与运行时快照写出
- `split.py`：数据索引与数据集划分封装
- `preprocess.py`：统计量、窗口、掩码、残差特征和预计算样本准备
- `model.py`：对外导出的 `EssenceForgeTCN`
- `features.py`：FFT 幅值构造与分支池化辅助函数
- `core/`：训练、微调、评估、数据集、损失函数和模型组件
- `configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json`：当前内置、可直接阅读的示例配置
- `environment.yml`：Conda 环境依赖列表

## 从零开始使用

### 0. 数据可用性说明

当前仓库没有上传原始实验数据。完整复现实验至少还需要：

- 使用官方工具处理后的遥测 CSV 数据，且目录结构与命名满足索引逻辑；
- 配置文件中的 `paths.data_dir` 指向该数据目录；
- 可写的缓存目录和输出目录。

如果没有原始数据：

- 可以正常阅读和修改代码；
- 可以创建 `conda` 环境；
- 可以查看配置和方法说明；
- 不能直接执行 `split` 和 `preprocess`；
- 也不能继续执行依赖这些产物的 `train`、`finetune` 和 `eval`。

### 1. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate essence-forge
```

`environment.yml` 默认安装 CPU 版 PyTorch。如果你要在 NVIDIA GPU 上训练，建议在激活环境后替换为 CUDA 版，例如：

```bash
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2. 准备数据

把数据集放到本机可访问的位置，然后编辑配置文件：

`configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json`

至少需要检查这些字段：

```json
{
  "paths": {
    "project_dir": ".",
    "data_dir": "D:/path/to/your/dataset",
    "outputs_dir": "./outputs"
  },
  "cache": {
    "enabled": true,
    "cache_dir": "D:/path/to/your/cache"
  },
  "train": {
    "device": "cuda",
    "use_amp": true
  }
}
```

说明：

- `paths.data_dir`：官方工具处理后的 CSV 数据根目录；
- `cache.cache_dir`：CSV 解析后的离线缓存目录，首次运行会在这里写入缓存；
- `paths.outputs_dir`：实验输出目录；
- `train.device`：有 GPU 用 `cuda`，无 GPU 改成 `cpu`；
- `train.use_amp`：如果使用 CPU，建议改成 `false`。

当前仓库内置配置包含作者本机的绝对路径，换机器后必须先改这些路径。

补充说明：

- `split` 会在 `paths.data_dir` 下递归查找 `Case_*.csv` 文件；
- `preprocess` 依赖 `split` 生成的 `split_source_*.json` 和 `split_target_*.json`；
- `train` / `finetune` 还依赖 `preprocess` 生成的 `source_stats.json` 与 `precomputed/` 产物。

### 3. 运行完整流水线

在仓库根目录执行：

```bash
python run.py pipeline \
  --config configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json \
  --run-dir outputs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s
```

如果你修改了数据、切窗配置、通道定义，或者想强制重建预处理产物，可以加上 `--force-rebuild`：

```bash
python run.py pipeline \
  --config configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json \
  --run-dir outputs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s \
  --force-rebuild
```

### 4. 分步执行

如果你想逐步排查流程，也可以拆开运行：

```bash
python run.py split --config configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json --run-dir outputs/debug_run
python run.py preprocess --config configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json --run-dir outputs/debug_run
python run.py train --config configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json --run-dir outputs/debug_run
python run.py finetune --config configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json --run-dir outputs/debug_run
python run.py eval --config configs/simplified_fft_lwpt_se_hil_a2b0_to_a2b1_stage3s.json --run-dir outputs/debug_run
```

## 运行产物

一个典型的 `run_dir` 下会看到这些内容：

- `split_source_*.json` / `split_target_*.json`：数据划分结果
- `source_stats.json`：源域归一化统计量
- `precomputed/`：预计算样本缓存
- `checkpoints/`：源域训练和目标域微调的模型权重
- `eval/`：评估指标、混淆矩阵、迁移差距汇总
- `eval/essence_forge_summary.json`：最终汇总文件

## 额外说明

- 仓库本身不依赖外部 `uav_tl` 或 `main.py`，但完整运行仍依赖仓库外部的原始数据集；
- 如果后续启用 targeted GAN 增强，需要同步检查配置中的 `augment.targeted_gan.*` 路径与设备字段；
- `outputs/` 默认被 `.gitignore` 忽略，适合作为实验产物目录。
