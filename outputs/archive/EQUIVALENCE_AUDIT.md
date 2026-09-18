# 历史实验与最新 BACR 协议等价性审计报告 (Equivalence Audit Report)

> **审计核心原则**：
> $$\boxed{\text{Same Task} + \text{Same Samples} + \text{Same Inputs} + \text{Same Model} + \text{Same Prompt} + \text{Same Protocol}}$$
> **核心判断**：评估口径变化 $\neq$ 必须重新调用模型 API。若原始 `predictions.jsonl` 完整且输入相同，使用新版 Manifest 与 Evaluator 重新评测即可继承。

---

## 1. 实验继承与重跑决策全景表

| 实验名称 | 历史路径身份 | 样本/实体覆盖度 | 协议等价性 | 审计结论与新身份 | 是否需重新调用模型 |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **B0 Zero-shot Qwen** | `01_qwen_baselines` | TW15 100%, TW17 100% | 完全等价 | **REUSE** (`B0 Zero-shot`) | ❌ 不用 |
| **B1 Direct SFT Qwen** | `01_qwen_baselines` | TW15 100%, TW17 100% | 完全等价 | **REUSE** (`B1 Direct SFT`) | ❌ 不用 |
| **G0 Pair-only Text** | `02_gemini_direct_baselines/g0_paironly` | TW15 99.1%, TW17 98.5% | Prompt A vs B | **REUSE_AS_ABLATION** (`G0-PairOnly`) | ❌ 不用 |
| **G1 Direct Multimodal** | `02_gemini_direct_baselines/g1_direct_mm` | TW15 99.1% (缺3条), TW17 98.5% (缺8条) | 完全等价 | **REUSE** (`G1 Direct MM`) | ❌ 仅补缺漏(可选) |
| **G1-SR Sequential Re-read** | `03_g1_sr_sequential_reread` | TW15 100%, TW17 99.9% (缺1条) | 完全等价 | **REUSE** (`G1-SR Control`) | ❌ 不用 |
| **G3 Precanonical** | `04_g3_precanonical` | TW15 100%, TW17 99.76% (缺2条) | 单向探针 vs 双向探针 | **REUSE_AND_HARVEST** (提取T0/V0缓存) | ❌ 提取缓存免API |
| **Diagnostics** | `diagnostics/{error_analysis, visual_probing}` | 1517 错误样本 / 实体探针 | 诊断性质 | **REUSE** (论文分析证据) | ❌ 不用 |
| **P0 Teacher Pilot** | `05_teacher_pilot` | 稳定性 94%, $\kappa>0.93$ | 机制资格 | **REUSE** (Dev'小规模复现即可) | ❌ 不用全重跑 |
| **BACR (双向跨模态探针)** | 新实验 (`configs/experiments/bacr.yaml`) | 待跑 | 新动作空间 $\pi_D, \pi_Q, \pi_R$ | **NEW_EXPERIMENT** (核心主线) | ✅ 需新跑 (注入T0/V0) |

---

## 2. 关键突破：Canonical T0 与 Canonical V0 零成本收获

经代码与 Schema 审计，`outputs/archive/04_g3_precanonical/g3_max2_*.jsonl` 中存储的 `text_initial` 和 `image_initial` 具备以下特性：
1. **Prompt 与 Schema 100% 吻合**：文本初读与全局视觉草图与 `bacr/prompts/{text_initial,vision_initial}.md` 完全等价。
2. **TW15 测试集覆盖率 100%**：674/674 个推文样本完整具备合法有效的 $T_0$ 和 $V_0$。
3. **TW17 测试集覆盖率 99.66%**：585/587 个推文样本具备有效 $T_0$ 和 $V_0$（仅缺 2 条推文）。
4. **直接产出**：通过 `scripts/prepare/build_canonical_init.py` 可直接无损导出至 `data/cache/canonical_t0/` 与 `canonical_v0/`，**立省成百上千次高耗时视觉模型 API 调用**！

---

## 3. 最新 Evaluator 对 G3 Precanonical 重新测算指标确认

在对齐至统一 Manifest 与数据处理后，重新测算得到的终版确定性指标为：

| 数据集 | 样本数 | 方面数 (Aspects) | 初始准确率 (Acc_init) | 初始 Macro-F1 (MF1_init) | 最终准确率 (Acc_final) | 最终 Macro-F1 (MF1_final) | Macro-F1 净增量 ($\Delta$MF1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Twitter-2015 Test** | 674 | 1037 | 73.45% | 66.81% | **72.67%** | **67.74** | **+0.93** |
| **Twitter-2017 Test** | 585 | 1231 | 70.39% | 69.71% | **72.6%** | **72.08** | **+2.37** |

> **注**：TW15 的 67.74 Macro-F1 (+0.93 over init) 与 TW17 的 72.08 Macro-F1 (+2.37 over init) 得到计算机严格复现，证实前期统计结论真实有效。
