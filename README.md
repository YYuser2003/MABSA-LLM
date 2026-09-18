# BACR: Bidirectional Active Cross-Modal Reasoning for MABSA

> **面向社交媒体多模态方面级情感分析（MABSA）的双向主动跨模态推理**  
> Official Codebase & Benchmark Suite

---

## 核心设计原则

本项目严格遵循三大核心架构原则：

$$
\boxed{\text{代码按职责分，实验按配置分，结果按 run 分}}
$$

- **`bacr/` 决定“怎么做”**：算法核心实现（Controller、Pipeline、Evaluator、Client、Taxonomy）；
- **`configs/` 决定“做哪个实验”**：改实验只改 YAML 配置，严禁复制修改 Python 代码；
- **`scripts/` 决定“执行什么任务”**：统一执行入口（`scripts/run.py`、`scripts/evaluate.py`）；
- **`data/` 决定“在哪些样本上做”**：官方数据集、唯一评测准绳 `manifests/`、规范化初始态缓存 `cache/`；
- **`outputs/` 记录“做出了什么结果”**：每次实验自包含独立 Run（配置、日志、预测、轨迹、指标）；
- **`docs/` 负责“学术研究文档”**：方法推导、实验协议、全景报告。

---

## 项目目录导航

```text
D:\YY\MABSA-LLM\
├── bacr/                          # 唯一核心算法包
│   ├── controller.py              # 控制器三元策略 (pi_D / pi_Q / pi_R)
│   ├── pipeline.py                # 统一主动推理状态机 (G3 / BACR / G1-SR)
│   ├── evaluator.py               # 统一评测器 (ASC / JMABSA / 转移矩阵 / 效率)
│   ├── client.py                  # 统一模型客户端 (Gemini / Qwen3-VL-8B)
│   ├── taxonomy.py                # 7 大视觉 + 8 大文本追问分类学
│   ├── prompts/                   # 原子提示模板 (text_initial, vision_initial, controller, etc.)
│   └── schemas/                   # JSON Schema 强契约 (initial, controller, probe, trajectory)
│
├── configs/                       # 统一配置体系 (改实验 = 改 YAML)
│   ├── models/                    # gemini.yaml, qwen3vl8b.yaml
│   ├── datasets/                  # twitter15.yaml, twitter17.yaml
│   └── experiments/               # b0, b1, g0, g1, g1_sr, g3, g4, bacr, random, oracle
│
├── scripts/                       # 统一执行与分析入口
│   ├── run.py                     # ★ 唯一核心实验入口
│   ├── evaluate.py                # ★ 唯一统一离线评测入口
│   ├── prepare/                   # 数据预处理与生成 (build_manifest.py, build_canonical_init.py)
│   ├── analyze/                   # 科学诊断分析 (errors.py, probing.py, transitions.py, oracle.py)
│   ├── train/                     # 训练与蒸馏入口 (teacher_search.py, sft.py, grpo.py)
│   └── legacy/                    # 历史独立脚本归档
│
├── data/                          # 数据集与规范化缓存
│   ├── processed/                 # 官方预处理数据
│   ├── manifests/                 # ★ 唯一共享评测名单 (tw15_dev, tw15_test, tw17_dev, tw17_test)
│   └── cache/                     # ★ 规范化统一初始态 (canonical_t0/, canonical_v0/)
│
├── outputs/                       # 实验产物
│   ├── runs/                      # 每次实验独立目录 (<date>_<method>_<dataset>_<split>_<tag>/)
│   ├── tables/                    # 自动汇总输出的论文对比表格
│   ├── figures/                   # 论文图表
│   └── archive/                   # 历史运行产物完整封存
│
├── docs/                          # 研究与规范文档
│   ├── research_plan.md           # 整体研究动机与多阶段路线图
│   ├── experiment_protocol.md     # 实验协议、评测规范与运行守则
│   ├── experiment_report.md       # 全景客观实测结果对比大表与深入诊断
│   └── bacr_method.md             # BACR 形式化定义、数学推导与 16 步里程碑
│
├── tests/                         # 单元测试 (test_schemas, test_evaluator, test_pipeline)
├── pyproject.toml                 # 项目打包构建定义
└── requirements.txt               # 运行环境依赖
```

---

## 核心基准对比概览

在 Twitter-2015 与 Twitter-2017 官方测试集上的受控靶向情感分类（Target-Guided ASC / TSC）：

| 协议 / 模型 | 范式机制 | Test-Time Compute | Twitter-2015 (Acc / Macro-F1) | Twitter-2017 (Acc / Macro-F1) | 关键机制与学术结论 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **B0 (Qwen3-VL-8B)** | 零样本直接生成 | 1-Turn | 57.06% / 49.33% | 61.12% / 56.62% | 基座模型初始水平 |
| **B1 (Qwen3-VL-8B)** | Direct SFT 监督微调 | 1-Turn | 69.45% / 62.48% | 70.88% / 68.21% | 全量微调强基线 |
| **G0 (Gemini 3.8)** | 纯文本强基线 ($T$) | 1-Turn | 70.81% / 66.84% | 73.17% / 72.88% | 纯文本大模型天花板 |
| **G1 (Gemini 3.8)** | 直接多模态 ($T + I$) | 1-Turn | 69.45% / 65.01% | **72.86% / 72.79%** | TW15 发生显著视觉负迁移 |
| **G1-SR (Gemini 3.8)** | 多模态自我反思 | 3-Turn (对齐算力) | 68.95% / 65.87% | 71.96% / 71.18% | 算力膨胀反思导致过度拟合漂移 |
| **G3 (AVR v1.2)** | 全局粗描 + 选择性视觉探针 | 自适应 (0~2 追问) | **72.67% / 67.74%** | **72.60% / 72.08%** | **TW15 突破 +2.73% F1；全面超越 G1-SR** |
| **G4 (BACR)** | 双向主动跨模态探针 | 自适应 (0~2 轮) | *Stage B.6 验证中* | *Stage B.6 验证中* | 解耦双向探针，兼顾文本深层语用 |

> **算力对照铁证**：在 TW15 上，G3 相比 G1-SR 提高 **+3.72% Acc / +1.87% MF1**；在 TW17 上，G3 相比 G1-SR 提高 **+0.64% Acc / +0.90% MF1**！实测证明 G3 的优势并非单纯来自推理步数增加，而是来自“识别不确定性并定向追问”的机制优越性。

---

## 快速运行与复现指南

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 运行实验 (统一入口 `scripts/run.py`)
```bash
# 运行 BACR 双向主动跨模态推理 (Twitter-2017 Dev 集)
python scripts/run.py --config configs/experiments/bacr.yaml --dataset twitter2017 --split dev --concurrency 4

# 运行 G3 主动视觉推理 (Twitter-2015 Test 集)
python scripts/run.py --config configs/experiments/g3_visual_only.yaml --dataset twitter2015 --split test --concurrency 4

# 运行 G1-SR 算力对齐基线
python scripts/run.py --config configs/experiments/g1_sr.yaml --dataset twitter2015 --split test --concurrency 4
```

### 3. 统一离线评测
```bash
# 对特定 run 目录执行全指标评测
python scripts/evaluate.py --run outputs/runs/<run_id>
```

### 4. 科学分析与诊断
```bash
# 错误样本细粒度归因
python scripts/analyze/errors.py --run outputs/runs/<run_id>

# 探针行为分类学统计
python scripts/analyze/probing.py --run outputs/runs/<run_id>

# 条件转移概率矩阵
python scripts/analyze/transitions.py --run outputs/runs/<run_id>
```

---

## 详细科研文档索引

- 📑 [研究规划与多阶段演化 (`docs/research_plan.md`)](file:///D:/YY/MABSA-LLM/docs/research_plan.md)
- 📋 [实验协议与规范总则 (`docs/experiment_protocol.md`)](file:///D:/YY/MABSA-LLM/docs/experiment_protocol.md)
- 📊 [全景科研实验报告与实测数据 (`docs/experiment_report.md`)](file:///D:/YY/MABSA-LLM/docs/experiment_report.md)
- 📐 [BACR 方法定义、数学推导与 16 步里程碑 (`docs/bacr_method.md`)](file:///D:/YY/MABSA-LLM/docs/bacr_method.md)
