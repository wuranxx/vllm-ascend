# Qwen3.5-9B W4A4 伪量化评测指南

本文档说明如何使用 vllm-ascend 的 fake_mx 伪量化方案对 Qwen3.5-9B 进行 W4A4 量化评测。
所有配置文件、脚本和转换工具均在本目录中，与 `vllm_ascend/quantization/methods/fake_mx.py` 代码保持一致。

## 目录结构

```
w4a4_eval/
├── README.md                          # 本文档
├── quick-verify-guide.md              # 快速验证方案（Level 子集）
├── configs/                           # 16 个量化配置文件
│   ├── qwen3_5_9b_rtn_attn-only_w4a4.json
│   ├── qwen3_5_9b_rtn_mlp-only_w4a4.json
│   ├── qwen3_5_9b_rtn_attn-mlp_w4a4.json
│   ├── qwen3_5_9b_rtn_attn-only_w4a4-w8a8-mixed.json
│   ├── qwen3_5_9b_rht_attn-only_w4a4.json
│   ├── qwen3_5_9b_rht_mlp-only_w4a4.json
│   ├── qwen3_5_9b_rht_attn-mlp_w4a4.json
│   ├── qwen3_5_9b_rht_attn-only_w4a4-w8a8-mixed.json
│   ├── qwen3_5_9b_flatquant_attn-only_w4a4.json
│   ├── qwen3_5_9b_flatquant_mlp-only_w4a4.json
│   ├── qwen3_5_9b_flatquant_attn-mlp_w4a4.json
│   ├── qwen3_5_9b_flatquant_attn-only_w4a4-w8a8-mixed.json
│   ├── qwen3_5_9b_lht_attn-only_w4a4.json
│   ├── qwen3_5_9b_lht_mlp-only_w4a4.json
│   ├── qwen3_5_9b_lht_attn-mlp_w4a4.json
│   └── qwen3_5_9b_lht_attn-only_w4a4-w8a8-mixed.json
├── scripts/
│   ├── serve/vllm_serve.sh            # vllm serve 启动脚本
│   ├── eval/run_math500.py            # MATH-500 评测脚本
│   ├── ptq/
│   │   ├── extract_ptq_data.sh        # 校准数据提取
│   │   └── run_amct_ptq.sh            # PTQ 多卡训练
│   └── convert/convert_ptq_to_vllm.py # PTQ 参数转换工具
```

## 配置文件命名规则

```
qwen3_5_9b_{algo}_{scope}_{precision}.json
```

| 字段 | 含义 | 取值 |
|------|------|------|
| algo | 量化算法 | rtn, rht, flatquant, lht |
| scope | 量化模块范围 | attn-only, mlp-only, attn-mlp |
| precision | 精度配置 | w4a4, w4a4-w8a8-mixed |

**w4a4**: 所有量化的 linear 层权重和激活均为 W4A4 MXFP4。
**w4a4-w8a8-mixed**: attention input 投影（qkv_proj/in_proj）为 W4A4 MXFP4，attention output 投影（o_proj/out_proj）为 W8A8 MXFP8，MLP 保持浮点。

## 量化算法说明

### RTN（Round-To-Nearest）

最简单的量化方式，直接对权重和激活做 MXFP4 QDQ（量化-反量化），无任何变换。

- 配置项：无额外配置
- scheme 名：`W4A4_MXFP4_FAKE`
- 代码位置：`_AscendFakeMXLinearMethod`
- 无需 PTQ 训练，无需外部参数

### RHT（Randomized Hadamard Transform）

在量化前对权重和激活施加随机 Hadamard 变换，打散离群值，降低量化误差。

- 配置项：
  - `rht_seed` — 随机 sign 序列的种子（默认 0）
  - `rht_group_size` — Hadamard 分块大小（默认同 group_size）
- scheme 名：`W4A4_MXFP4_RHT_FAKE` / `W8A8_MXFP8_RHT_FAKE`
- 代码位置：`_AscendRHTFakeMXLinearMethod`
- 无需 PTQ 训练，用 seed 生成随机 signs
- 权重变换：运行时自动旋转权重（`process_weights_after_loading` 中调用 `randomized_hadamard_transform`）
- 激活变换：`randomized_hadamard_transform(x, signs, group_size)`

### FlatQuant

学习型 Kronecker 变换，通过 PTQ 训练优化 left/right 变换矩阵和 diag_scale。

- 配置项：
  - `flatquant_params_path` — 外部参数文件路径（相对于模型目录）
  - `max_supported_tp` — 最大支持的 TP 数（默认 4）
  - `flatquant_matrix_size` — AMCT 分解的矩阵大小 K（默认 128）
  - `flatquant_use_diag` — 是否使用 diag_scale（默认 true）
  - `group_size` — MX 分块大小（默认 32）
- scheme 名：`W4A4_MXFP4_FLATQUANT_FAKE` / `W8A8_MXFP8_FLATQUANT_FAKE`
- 代码位置：`_AscendFakeMXFlatQuantLinearMethod`
- 需要 PTQ 训练生成参数
- 权重变换：`W' = inv(left) @ reshape(W) @ inv(right).T / diag_scale`
- 激活变换：`x' = reshape(left.T @ reshape(x) @ right) * diag_scale`
- 参数文件格式（safetensors）：
  ```
  {prefix}.weight           — 变换后的 FP 权重 [out, in]
  {prefix}.left_trans       — 左变换矩阵 [L, L]
  {prefix}.right_trans      — 右变换矩阵 [R, R]
  {prefix}.clip_ratio       — 裁剪比例 [1] (float32, 值=1.0)
  {prefix}.diag_scale       — 对角缩放 [L*R] (float32, 可选)
  ```
  其中 L*R = in_features，L 和 R 由 AMCT 训练确定

### LHT（Learnable Hadamard Transform）

学习型 Hadamard 变换，通过 PTQ 训练优化一个 K×K 的正交变换矩阵。

- 配置项：
  - `lht_params_path` — 外部参数文件路径（相对于模型目录，**必填**）
  - `hadamard_learning_matrix_size` — 变换矩阵大小 K（默认 128）
  - `group_size` — MX 分块大小（默认 32）
- scheme 名：`W4A4_MXFP4_HADAMARD_LEARNING_FAKE` / `W8A8_MXFP8_HADAMARD_LEARNING_FAKE`
- 代码位置：`_AscendHadamardLearningFakeMXLinearMethod`
- 需要 PTQ 训练生成参数
- 权重变换：`W' = reshape(W, -1, K) @ inv(T).T`
- 激活变换：`x' = reshape(x, -1, K) @ T`
- 参数文件格式（safetensors）：
  ```
  {prefix}.transform_weight — K×K 变换矩阵 [K, K] (bfloat16)
  ```
  其中 K = hadamard_learning_matrix_size（默认 128），in_features 必须能被 K 整除

## 模块覆盖说明

配置通过 `module_quant_overrides` 的 glob 模式匹配模块路径。匹配规则使用 `fnmatchcase`，按字典序遍历，**先匹配到的 pattern 生效**。

Qwen3.5-9B 的关键模块路径：
```
model.language_model.layers.{N}.self_attn.qkv_proj.weight      # attention input (fused QKV)
model.language_model.layers.{N}.self_attn.o_proj.weight         # attention output
model.language_memory.layers.{N}.linear_attn.in_proj_qkvz.weight  # GDN input (fused QKVZ)
model.language_memory.layers.{N}.linear_attn.in_proj_ba.weight    # GDN input (fused BA)
model.language_memory.layers.{N}.linear_attn.out_proj.weight      # GDN output
model.language_memory.layers.{N}.mlp.gate_proj.weight           # MLP gate
model.language_memory.layers.{N}.mlp.up_proj.weight             # MLP up
model.language_memory.layers.{N}.mlp.down_proj.weight           # MLP down
```

> **注意**：`self_attn` 和 `linear_attn` 在不同层交替出现（Qwen3.5 混合架构），`*self_attn*` 和 `*linear_attn*` 的 glob 会分别匹配。

所有配置都有 `"*": "FLOAT"` 作为兜底，确保未显式指定的模块（embed_tokens、lm_head、visual、MTP 等）保持浮点。

## 使用步骤

### 1. RTN / RHT 评测（无需 PTQ 训练）

```bash
# 复制配置到模型目录
cp configs/qwen3_5_9b_rtn_attn-only_w4a4.json /path/to/model/quant_model_description.json

# 启动 vllm serve
# RTN 不需要 --enforce-eager；RHT/FlatQuant/LHT 建议加 --enforce-eager
./scripts/serve/vllm_serve.sh /path/to/model 0 8001 configs/qwen3_5_9b_rtn_attn-only_w4a4.json

# 运行评测
python scripts/eval/run_math500.py 8001 ./outputs/rtn_attn-only_w4a4
```

### 2. FlatQuant / LHT 评测（需要 PTQ 训练）

#### 2.1 提取校准数据

```bash
# attn 层校准数据
./scripts/ptq/extract_ptq_data.sh /path/to/model /data/ptq_data attn-linear 0

# mlp 层校准数据
./scripts/ptq/extract_ptq_data.sh /path/to/model /data/ptq_data mlp 1
```

#### 2.2 PTQ 训练（8 卡并行）

```bash
# FlatQuant attn 训练
./scripts/ptq/run_amct_ptq.sh flatquant attn-linear /path/to/model /data/ptq_data /data/ptq_fq_attn

# LHT attn 训练
./scripts/ptq/run_amct_ptq.sh learnable_had attn-linear /path/to/model /data/ptq_data /data/ptq_lht_attn

# FlatQuant mlp 训练
./scripts/ptq/run_amct_ptq.sh flatquant mlp /path/to/model /data/ptq_data /data/ptq_fq_mlp

# LHT mlp 训练
./scripts/ptq/run_amct_ptq.sh learnable_had mlp /path/to/model /data/ptq_data /data/ptq_lht_mlp
```

#### 2.3 转换参数

```bash
# FlatQuant attn 参数（需要模型权重做权重变换）
python scripts/convert/convert_ptq_to_vllm.py \
  --algo flatquant --target attn-linear \
  --ptq_dir /data/ptq_fq_attn/ptq_params/qwen3_5/attn-linear \
  --model_dir /path/to/model \
  --output /data/flatquant_attn_params.safetensors

# FlatQuant mlp 参数
python scripts/convert/convert_ptq_to_vllm.py \
  --algo flatquant --target mlp \
  --ptq_dir /data/ptq_fq_mlp/ptq_params/qwen3_5/mlp \
  --model_dir /path/to/model \
  --output /data/flatquant_mlp_params.safetensors

# 合并 attn + mlp 参数（用于 attn-mlp 场景）
python scripts/convert/convert_ptq_to_vllm.py \
  --merge /data/flatquant_attn_params.safetensors \
          /data/flatquant_mlp_params.safetensors \
          /data/flatquant_attn_mlp_params.safetensors

# LHT attn 参数（不需要模型权重）
python scripts/convert/convert_ptq_to_vllm.py \
  --algo lht --target attn-linear \
  --ptq_dir /data/ptq_lht_attn/ptq_params/qwen3_5/attn-linear \
  --output /data/lht_attn_params.safetensors

# LHT mlp 参数
python scripts/convert/convert_ptq_to_vllm.py \
  --algo lht --target mlp \
  --ptq_dir /data/ptq_lht_mlp/ptq_params/qwen3_5/mlp \
  --output /data/lht_mlp_params.safetensors

# 合并 LHT attn + mlp
python scripts/convert/convert_ptq_to_vllm.py \
  --merge /data/lht_attn_params.safetensors \
          /data/lht_mlp_params.safetensors \
          /data/lht_attn_mlp_params.safetensors
```

#### 2.4 部署参数并评测

```bash
# 将参数文件放到模型目录（或配置中指定的路径）
cp /data/flatquant_attn_params.safetensors /path/to/model/flatquant_params.safetensors

# 启动 vllm serve（FlatQuant/LHT 必须加 --enforce-eager）
./scripts/serve/vllm_serve.sh /path/to/model 0 8001 \
  configs/qwen3_5_9b_flatquant_attn-only_w4a4.json --enforce-eager

# 评测
python scripts/eval/run_math500.py 8001 ./outputs/flatquant_attn-only_w4a4
```

## 配置项与代码对照表

| 配置 JSON key | 代码读取位置 | 说明 |
|--------------|------------|------|
| `default_quant_type` | `modelslim_config.py` | 默认 scheme 名 |
| `module_quant_overrides` | `modelslim_config.py` | glob 模式覆盖 |
| `group_size` | `fake_mx.py` `_AscendFakeMXLinearMethod.__init__` | MX 分块大小 |
| `fake_mx_quant_targets` | `fake_mx.py` `_AscendFakeMXLinearMethod.__init__` | 额外非 Linear 节点列表；当前仅支持 `attn-cache` |
| `rht_seed` | `fake_mx.py` `_AscendRHTFakeMXLinearMethod.__init__` | RHT 随机种子 |
| `rht_group_size` | `fake_mx.py` `_AscendRHTFakeMXLinearMethod.__init__` | RHT 分块大小 |
| `flatquant_params_path` | `fake_mx.py` `_AscendFakeMXFlatQuantLinearMethod.__init__` | FlatQuant 参数文件路径 |
| `max_supported_tp` | `fake_mx.py` `_AscendFakeMXFlatQuantLinearMethod.__init__` | FlatQuant 最大 TP |
| `flatquant_matrix_size` | `fake_mx.py` `_AscendFakeMXFlatQuantLinearMethod.__init__` | FlatQuant AMCT 矩阵大小 K |
| `flatquant_use_diag` | `fake_mx.py` `_AscendFakeMXFlatQuantLinearMethod.__init__` | 是否使用 diag_scale |
| `lht_params_path` | `fake_mx.py` `_AscendHadamardLearningFakeMXLinearMethod.__init__` | LHT 参数文件路径（必填） |
| `hadamard_learning_matrix_size` | `fake_mx.py` `_AscendHadamardLearningFakeMXLinearMethod.__init__` | LHT 矩阵大小 K |

> **注意**：重构后 `fake_mx_weight_state`、`auto_transform`/`auto_rotate` 系列配置项已移除。变换始终在加载时自动执行，无需手动开关。

## 评测结果（Qwen3.5-9B MATH-500）

### 全 W4A4 矩阵

| 算法 | attn-only | mlp-only | attn-mlp |
|------|-----------|----------|----------|
| RTN | 86.20% | 92.60% | 77.40% |
| RHT | 89.20% | 90.60% | 79.60% |
| FlatQuant | 92.40% | 94.40% | 91.20% |
| LHT | 93.00% | 93.00% | 92.80% |

BF16 基线：93.80%

### Mixed W4A4/W8A8（attn-only）

| RTN | RHT | FlatQuant | LHT |
|------|-----|-----------|-----|
| 89.80% | 91.60% | 92.60% | 93.00% |

### 重构后 Level3 快速验证（2026-08-11）

| 算法 | 全量 Level3 | 重构后 Level3 | 差值 | 状态 |
|------|------------|-------------|------|------|
| LHT | 97.14% | 97.14% | 0% | ✅ 一致 |
| FlatQuant | 97.14% | 99.05% | +1.91% | ✅ 无回归 |

### 关键结论

1. **LHT 在所有场景均最优或并列最优**，attn-only 全 W4A4 达到 93.0%（仅 -0.8% vs BF16）
2. **FlatQuant mlp-only 94.4% 超越 BF16 基线**，MLP 量化后精度反而提升
3. **attn 是 W4A4 的主要损失来源**：RTN attn-only -7.6% vs mlp-only -1.2%
4. **学习型变换（FlatQuant/LHT）收益巨大**：attn-mlp 场景 RTN 77.4% → LHT 92.8%（+15.4%）
5. **RHT 对 mlp 反而有负面影响**：RTN mlp 92.6% → RHT mlp 90.6%（-2.0%），随机变换引入噪声

## 环境要求

- vllm-ascend 分支：`w4a4-quant-eval-results`
- CANN >= 9.0.1
- Ascend 910 NPU
- AMCT PTQ 训练需独立 conda 环境（amct_pytorch）
- evalscope >= 1.9.1
