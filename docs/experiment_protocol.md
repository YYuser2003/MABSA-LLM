# BACR Experiment Protocol & Standardization Specification

> **实验规范原则**:
> 
> $$
> \boxed{\text{代码按职责分，实验按配置分，结果按 run 分}}
> $$

---

## 一、配置驱动运行机制

所有的实验差异由 `configs/` 下的 YAML 配置文件定义，严禁为新实验复制或新建 Python 脚本。

### 运行命令规范
```bash
# 标准语法
python scripts/run.py --config configs/experiments/<EXP_NAME>.yaml --dataset <DATASET> --split <SPLIT> [OPTIONS]

# 示例：运行 BACR 协议在 Twitter-2017 Dev 集上
python scripts/run.py --config configs/experiments/bacr.yaml --dataset twitter2017 --split dev --concurrency 4

# 示例：运行 G3 主动视觉推理在 Twitter-2015 Test 集上
python scripts/run.py --config configs/experiments/g3_visual_only.yaml --dataset twitter2015 --split test --concurrency 4
```

---

## 二、运行标识符规范 (Run ID Schema)

每次实验的运行目录自动命名为：
```text
<date>_<method>_<dataset>_<split>_<tag>
```

例如：
- `20260918_143000_bacr_twitter2017_dev`
- `20260918_150000_g3_visual_only_twitter2015_test_seed42`

每个 Run 目录自包含以下完整资产，确保 100% 可独立复现：
```text
outputs/runs/<run_id>/
├── config.yaml          # 解析后的全局配置快照 (包含 CLI 参数、模型超参、数据集路径)
├── predictions.jsonl    # 最终用于评测的极性预测文件
├── trajectories.jsonl   # 完整的多轮思考、追问、回答与决策轨迹
├── metrics.json         # 自动评测生成的全维度指标报告
└── run.log              # 运行全生命周期的时间戳日志
```

---

## 三、Manifest 与 Canonical Cache 规约

### 1. 唯一实验名单 (`data/manifests/`)
所有实验必须严格对齐 `data/manifests/` 下的预处理名单：
- `tw15_dev.jsonl` (1122 aspects)
- `tw15_test.jsonl` (1037 aspects)
- `tw17_dev.jsonl` (1176 aspects)
- `tw17_test.jsonl` (1234 aspects)

不允许任何模型自行为推文抽取不同的 Aspect 数量，以此彻底消除“样本数量分母不一致”导致的统计偏置。

### 2. 统一初始态与规范实验阶梯 (`data/cache/`)
为了保证因果对比的绝对严谨性，体系严格确立：
$$
\boxed{\text{Text-only baseline} \equiv T_0 = \text{Text Initial Reasoning}}
$$
$$
\boxed{T_0 \longrightarrow S_0 = (T_0 + V_0) \longrightarrow \text{Probe} \longrightarrow F}
$$

- $T_0$（纯文本初始判断）：来自 `data/cache/canonical_t0/`；
- $V_0$（全局视觉初探）：来自 `data/cache/canonical_v0/`。

完整的实验阶梯映射表如下：

| 实验标识 | 完整规范名称 | 输入与流程定义 | 判定策略 | 对应配置文件 |
|---|---|---|---|---|
| **G0 / T0** | Text Initial Reasoner | $T_0$ (0 API Calls) | 直接取 $T_0.\text{pairs}$ 作为纯文本基线 | `configs/experiments/g0_t0.yaml` |
| **G1** | Direct Multimodal | Tweet + Raw Image | 端到端单步直接输出 Baseline | `outputs/archive/gemini_direct_mm/` |
| **S0** | Global Visual Sketch | $T_0 + V_0$, $k=0$ | Controller 综合 $T_0$ 与 $V_0$ 评估，无追问 | `configs/experiments/s0_sketch.yaml` |
| **G3** | Vision Deep Probe | $T_0 + V_0 + \text{Vision Probe}$ | $\pi_D \in \{\text{VISUAL}, \text{STOP}\}$ | `configs/experiments/g3_vision_probe.yaml` |
| **G4-TP** | Text Deep Probe | $T_0 + V_0 + \text{Text Probe}$ | $\pi_D \in \{\text{TEXT}, \text{STOP}\}$ | `configs/experiments/g4_tp.yaml` |
| **BACR** | Bidirectional Probe | $T_0 + V_0 + \text{Bi-Probe}$ | $\pi_D \in \{\text{VISUAL}, \text{TEXT}, \text{STOP}\}$ | `configs/experiments/bacr.yaml` |

所有方法严格继承同一组 $T_0$ 和 $V_0$，彻底消除初始状态抖动对对比评测造成的干扰。

---

## 四、核心评测指标体系

1. **ASC 指标**：
   - Accuracy (%)
   - Macro-F1 (%)
2. **Aspect-Locked 纯粹情感增益**：
   - $ERR_{\text{sent}}$: 错题救回率 ($W \to C$)
   - $HRR_{\text{sent}}$: 负向误修改率 ($C \to W$)
   - $VNG$: 净增益 ($ERR - HRR$)
3. **推断效率指标**：
   - Macro-F1 / API Call
   - Macro-F1 / 1K Tokens
   - Macro-F1 / Image Invocation
   - Average Latency (ms/aspect)
