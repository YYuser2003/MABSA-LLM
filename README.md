# BACR: Bidirectional Active Cross-Modal Reasoning for MABSA

> **Official Implementation of BACR-v3: Multimodal Aspect-Based Sentiment Analysis**  
> *Text Anchor + Risk Diagnosis + Evidence Firewall + Selective Deliberation*

---

## 1. Architectural Overview (BACR-v3)

Standard Multimodal Aspect-Based Sentiment Analysis (MABSA) often suffers from **visual noise contamination**, where irrelevant or decorative image elements overturn correct linguistic decisions. BACR-v3 addresses this failure mode by treating the text as an initial anchor, using global vision strictly on the symbolic control plane, and enforcing strict fail-closed evidence governance.

$$
\boxed{
H_A \parallel I_G \rightarrow C_D \rightarrow 
\left[
T_R \rightarrow C_T
\;\middle|\;
I_P \rightarrow C_E \rightarrow T_F \rightarrow C_V
\right]_{\le B_{\text{deep}}} 
\rightarrow Y_{\text{final}}
}
$$

### Single State Machine Principle

$$
\boxed{\text{Pipeline} = \text{BACREnv} + \text{ControllerPolicy} + \text{Logger}}
$$

All state transitions, fail-closed firewalls, aspect isolation, and verifier reversions are strictly centralized within `BACREnv.step()`. This guarantees that Teacher inference, Student evaluation, and Reinforcement Learning (GRPO) share **completely isomorphic transition dynamics**.

```text
                     ControllerPolicy (Teacher / Student / Rule / RL)
                                     │
                                     ▼
Sample (T, I) ──────────────► BACREnv.step() ──────────────► Standardized Trajectory
                                     │
                                     ▼
                         New Observable State S_{t+1}
```

---

## 2. Core Theoretical Invariants

$$
\boxed{\textbf{No verifier approval, no baseline mutation.}}
$$

1. **Dual Baseline Protection ($H_A$ & $H_B$)**:
   - $H_A$: Immutable Text Anchor Ledger.
   - $H_B$: Current Verified Text Baseline Ledger.
   - Verifier Rejection ($\boxed{\text{REVERT\_TEXT\_BASELINE}}$) immediately reverts any candidate hypothesis back to $H_B$.
2. **Information-Flow Firewall ($V_0 \not\to T_R$)**:
   - $C_D$ splits outputs into `text_risk` (syntactic/reporting frame) and `visual_opportunity` (face/object presence).
   - $C_Q^T$ receives purely textual risk with zero visual fields, physically barring $V_0$ from leaking into $T_R$.
3. **Fail-Closed Evidence Firewall ($C_E$)**:
   - All visual probes must pass verification: unparseable, invalid, unbound, or contradictory evidence automatically falls back to `INVALID` / `NON_DECISIVE`.
4. **Structured Evidence Bundle (`EvidenceBundle`)**:
   - Multi-round probes accumulate structured probe items with provenance citations (`ref: probe_1`). $T_F$ only sees certified facts.
5. **Monotonic Transition Ordering**:
   - Text transitions are recorded before visual escalation, guaranteeing strictly monotonic step ordering: $step_{t+1} > step_t$.
6. **Asymmetric Aspect-Level Reward**:
   - Step rewards penalize harmful mutations more heavily than missed recoveries ($\beta = 1.5 > \alpha = 1.0$) with compute cost penalties.
7. **Task Modes & Gold Sequestration**:
   - `target_guided`: Evaluates sentiment given target aspects.
   - `open_joint`: End-to-end extraction and classification where Gold aspects are strictly sequestered from inference states.

---

## 3. Directory Layout

```text
MABSA-LLM/
├── bacr/
│   ├── client.py            # Gemini & OpenAI-compatible / vLLM API clients
│   ├── env.py               # Unified BACREnv (Gym-like Aspect-Level MDP)
│   ├── evaluator.py         # Standardized pair F1 & Transition MU_T / MU_V evaluator
│   ├── meta_controller.py   # Meta-Controller (C_D, C_Q^T, C_T, C_E, C_V, C_A)
│   ├── pipeline_v3.py       # Thin Pipeline Runner over BACREnv + Policy
│   ├── policies/            # Controller Policies (GeminiTeacherPolicy, RulePolicy, etc.)
│   ├── prompts/             # System prompts for Reasoners, Sensors, Controllers
│   ├── schemas_v3.py        # Strict Pydantic v2 data contracts
│   ├── text_reasoner.py     # Text Anchor (T_A), Rethink (T_R), Fusion (T_F)
│   └── vision_sensor.py     # Global Visual Sketch (I_G), Deep Probe (I_P)
├── configs/
│   ├── datasets/            # Dataset definitions (Twitter-2015, Twitter-2017)
│   ├── experiments/         # Experiment YAML configurations
│   └── models/              # Model endpoint and hyperparameter configs
├── scripts/
│   ├── run.py               # Multi-worker experiment runner
│   └── evaluate.py          # Standalone evaluation entrypoint
├── tests/                   # Comprehensive test suite & invariant verifications
└── .github/workflows/ci.yml # Automated GitHub Actions CI
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
# BACR-v3 Full Run on Twitter-2015 test set
python scripts/run.py --config configs/experiments/bacr_v3.yaml --tag v3_production

# Run with sample limit for quick validation
python scripts/run.py --config configs/experiments/bacr_v3.yaml --limit 5 --tag smoke_test
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
