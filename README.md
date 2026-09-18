# BACR: Bidirectional Active Cross-Modal Reasoning for MABSA

> **Official Implementation of BACR-v3 Minimal Teacher Verification**  
> *Text Anchor + Risk Diagnosis + Evidence Firewall + Selective Deliberation*

---

## 1. Architectural Overview (BACR-v3 Minimal Teacher)

Standard Multimodal Aspect-Based Sentiment Analysis (MABSA) often suffers from **visual noise contamination**, where irrelevant or decorative image elements overturn correct linguistic decisions. BACR-v3 addresses this failure mode by anchoring on the text baseline ($T_0$), using global vision ($V_0$) strictly on the symbolic control plane, and enforcing strict fail-closed evidence governance.

The primary goal of the Minimal Teacher stage is strictly to verify the foundational hypothesis:

$$
\boxed{\text{BACR-v3 Teacher: } \text{Final} > T_0 \quad (\text{Recover} - \text{Harm} > 0)}
$$

### Minimal Teacher Pipeline Architecture ($K=1$)

$$
\boxed{
T_0 + V_0 + a_{\text{gold}} \longrightarrow C_R \longrightarrow 
\begin{cases}
\text{KEEP} & \longrightarrow \text{Final} = T_0 \\
\text{TEXT} & \longrightarrow T_R \longrightarrow C_F \longrightarrow \text{Final} \\
\text{VISION} & \longrightarrow I_P \longrightarrow C_E \longrightarrow T_F \longrightarrow C_F \longrightarrow \text{Final}
\end{cases}
}
$$

```text
                                  Gold Aspect (a_gold)
                                          │
                                          ▼
   Tweet Text (T_raw) ──► T0 Anchor ──► Meta-Controller (C_R) ◄── V0 Sketch ◄── Image (I)
                                          │
             ┌────────────────────────────┼───────────────────────────┐
             ▼                            ▼                           ▼
        [Route KEEP]                 [Route TEXT]               [Route VISION]
             │                            │                           │
         No Action               Linguistic Rethink (TR)       Targeted Probe (IP)
             │                       (No Image Access)                │
             │                            │                    Evidence Firewall (CE)
             │                            │                     (Fail-Closed Filter)
             │                            │                           │
             │                            │                 Evidence Fusion (TF)
             │                            │                           │
             │                            └───────────┬───────────────┘
             │                                        ▼
             │                              Final Audit Verifier (CF)
             │                                (Revert-to-T0 Guard)
             │                                        │
             ▼                                        ▼
       Final = T0 Baseline                     Final Prediction
```

---

## 2. Core Epistemic Invariants

$$
\boxed{\textbf{No verifier approval, no baseline mutation.}}
$$

1. **$K=1$ Single-Pass Intervention**:
   - Exactly one discrete route decision per gold aspect ($a_{\text{gold}}$). Multi-aspect tweets are factored into independent aspect-level evaluations and aggregated into `final_pairs`.
2. **Fail-Closed Evidence Firewall ($C_E$)**:
   - $$\boxed{\text{EvidenceInvalid} \Longrightarrow T_0}$$
   - If visual evidence returned by the sensor ($I_P$) is unparseable, occluded, unbound, or non-decisive, the pipeline immediately falls back to $T_0$ without invoking the cross-modal reasoner ($T_F$).
3. **Unified Final Audit Verifier ($C_F$)**:
   - $$\boxed{\text{AuditReject} \Longrightarrow T_0}$$
   - Any proposed candidate revision away from $T_0$ must be conclusively justified by explicit syntactic scope (TEXT route) or verified bound physical facts (VISION route). Speculative or unsupported revisions are strictly reverted to $T_0$.
4. **Information-Flow Firewall ($V_0 \not\to T_R$)**:
   - The global visual sketch ($V_0$) is strictly restricted to the Controller's routing decision ($C_R$) and never exposed to the text reasoner ($T_R$). Any accidental visual words in the Controller's critique are programmatically sanitized.
5. **Multi-Aspect Isolation**:
   - Non-target aspects in multi-aspect samples are strictly locked to their baseline sentiments during single-aspect evaluation, preventing cross-aspect affect drift.

---

## 3. Directory Layout

```text
MABSA-LLM/
├── bacr/
│   ├── client.py            # Gemini & OpenAI-compatible / vLLM API clients
│   ├── evaluator.py         # Standardized F1, Transition MU, and Teacher Verification Evaluator
│   ├── meta_controller.py   # Meta-Controller (decide_route, filter_evidence, audit_revision)
│   ├── pipeline.py          # Baseline G3 / BACR-v2 pipelines (legacy preservation)
│   ├── pipeline_v3.py       # Minimal Teacher Pipeline Runner (run_aspect, run_sample)
│   ├── prompts/v3/          # System prompts (controller_route, firewall, audit, rethink, fusion)
│   ├── schemas_v3.py        # Strict Pydantic v2 data contracts (RouteDecision, EvidenceResult, etc.)
│   ├── text_reasoner.py     # Deliberative Reasoner (rethink_text, fuse_evidence)
│   └── vision_sensor.py     # Objective Visual Sensor (perceive_global, probe_deep)
├── configs/
│   ├── datasets/            # Dataset definitions (Twitter-2015, Twitter-2017)
│   ├── experiments/         # Experiment YAML configurations (bacr_v3.yaml, bacr_v2.yaml)
│   └── models/              # Model endpoint configs
├── scripts/
│   ├── run.py               # Master experiment runner with resume & multi-worker pool
│   └── evaluate.py          # Standalone evaluation entrypoint
├── tests/                   # Comprehensive unit tests & invariant verifications
└── .github/workflows/ci.yml # Automated CI pipeline
```

---

## 4. Quick Start

### Installation

```bash
git clone https://github.com/YYuser2003/MABSA-LLM.git
cd MABSA-LLM
pip install pydantic>=2.0 pyyaml pillow requests
```

### Environment Configuration

Create a `.env` file in the root directory (gitignored):

```bash
GEMINI_API_KEY=your_gemini_api_key
GEMINI_BASE_URL=https://your-endpoint.com/v1
```

For local vLLM / Qwen3-VL-8B-Instruct inference:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=EMPTY
```

### Running Experiments

```bash
# BACR-v3 Minimal Teacher Verification on Twitter-2015 dev set (Recommended)
python scripts/run.py --config configs/experiments/bacr_v3.yaml --split dev --limit 20 --tag dev_pilot

# Full run on dev set
python scripts/run.py --config configs/experiments/bacr_v3.yaml --split dev --tag v3_dev_full
```

### Running Invariant Tests

```bash
python tests/test_schemas.py
python tests/test_evaluator.py
python tests/test_pipeline.py
python tests/test_pipeline_v3.py
```

---

## 5. Citation

If you find this codebase useful for your research, please cite:

```bibtex
@article{mabsa_bacr2026,
  title={BACR: Bidirectional Active Cross-Modal Reasoning for Multimodal Aspect-Based Sentiment Analysis},
  author={Lab Research Team},
  year={2026}
}
```
