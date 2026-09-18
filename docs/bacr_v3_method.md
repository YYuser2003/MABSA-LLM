# BACR-v3: Minimal Teacher Verification & Decision Governed Reasoning

> **Official Methodology Specification of BACR-v3 Minimal Teacher**  
> *Target-Guided Single-Pass Routing, Epistemic Evidence Firewall, and Safeguarded Decision Auditing*

---

## 1. Executive Summary & Foundational Hypothesis

Multimodal Aspect-Based Sentiment Analysis (MABSA) on social media poses a fundamental epistemic challenge: **visual noise and decorative imagery frequently induce visual hallucination, overturning correct linguistic judgements**.

BACR-v3 reformulates multimodal reasoning from unconstrained cross-modal fusion into **Decision-Governed Selective Deliberation**:
- Anchor solidly on the pure text baseline ($T_0$).
- Restrict global visual perceptions ($V_0$) strictly to the **symbolic control plane** (the Meta-Controller), with an information-flow barrier preventing uninspected imagery from reaching the text reasoner ($V_0 \not\to T_R$).
- Enforce a strict **Fail-Closed Evidence Firewall** ($C_E$) and a **Final Audit Safeguard** ($C_F$), guaranteeing:
  
$$\boxed{\textbf{No verifier approval, no baseline mutation.}}$$

### The Foundational Verification Hypothesis

Before training student models via SFT or Reinforcement Learning (PPO/GRPO), the Teacher must empirically prove positive marginal utility:

$$\boxed{\text{BACR-v3 Teacher Verification Hypothesis: } \text{Final} > T_0 \iff \text{Recover} - \text{Harm} > 0}$$

---

## 2. Minimal Teacher Architecture ($K=1$)

For a given tweet text $T$, image $I$, and gold aspect $a_{\text{gold}}$:

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

## 3. The 3 Discrete Routes

### Route 1: `KEEP` (Zero Action / Pure Anchor Passthrough)
- **Condition**: Controller diagnoses that the text anchor $T_0$ is already decisive, syntactically unambiguous, and unaffected by the visual context.
- **Action**: $\text{Final} = T_0$.
- **Cost**: 0 reasoning calls, 0 vision calls.
- **Transition Record**: Formally logged as an active `KEEP` transition to prevent selection bias in downstream SFT/RL training.

### Route 2: `TEXT` (Targeted Linguistic Rethink)
- **Condition**: Controller identifies text-internal risk: syntactic irony, negation inversion, multiple aspects with conflicting sentiments, or rhetorical contrast.
- **Step 2.1 (Critique Generation)**: Controller generates a purely linguistic critique $C_Q^T(a_{\text{gold}})$.
  - **Hard Invariant**: $V_0 \not\to T_R$. The prompt to the Text Reasoner never receives $V_0$. Any accidental perceptual vocabulary in the controller critique is programmatically sanitized by the semantic firewall.
- **Step 2.2 (Linguistic Deliberation)**: Text Reasoner evaluates the aspect under critique and proposes a candidate prediction:
  $$y_{\text{cand}} = T_R(T, a_{\text{gold}}, T_0, C_Q^T)$$
- **Step 2.3 (Final Audit)**: Verifier $C_F$ audits $y_{\text{cand}}$:
  - If accepted: $\text{Final} = y_{\text{cand}}$.
  - If rejected: $\text{Final} = T_0$ (fail-safe reversion).

### Route 3: `VISION` (Targeted Probe & Epistemic Firewall)
- **Condition**: Controller diagnoses an image-dependent sentiment conflict (e.g. text seems neutral or sarcastic, but image contains decisive facial expressions, text-on-image logos, or physical actions).
- **Step 3.1 (Targeted Question Formulation)**: Controller formulates a targeted, aspect-bound visual question $Q_V$.
  - **Fail-Closed Rule**: If the controller fails to output a non-empty question, the route immediately downgrades to `KEEP` without touching the vision sensor.
- **Step 3.2 (Visual Probe)**: Vision Sensor executes deep visual inspection:
  $$E_{\text{raw}} = I_P(I, Q_V)$$
- **Step 3.3 (Fail-Closed Evidence Firewall)**: Controller audits the raw evidence against strict epistemic criteria:
  
$$\boxed{\text{EvidenceResult} = C_E(a_{\text{gold}}, Q_V, E_{\text{raw}})}$$

The evidence is approved **if and only if** all five predicates hold:

$$\boxed{
\text{PassFirewall} \iff \begin{cases}
\text{status} = \text{VALID} \\
\land \quad \text{target\_binding} = \text{DIRECT} \\
\land \quad \text{relevance} = \text{HIGH} \\
\land \quad \text{revision\_support} = \text{SUPPORTS\_REVISION} \\
\land \quad |\text{usable\_evidence}| > 0
\end{cases}
}$$

If any condition fails, the pipeline immediately reverts to $T_0$ without invoking the multimodal fusion reasoner ($T_F$ bypassed).
- **Step 3.4 (Cross-Modal Evidence Fusion)**: If firewall passes, Text Reasoner fuses the text with verified physical evidence:
  $$y_{\text{cand}} = T_F(T, a_{\text{gold}}, T_0, E_{\text{verified}})$$
- **Step 3.5 (Final Audit)**: Verifier $C_F$ performs final coherence audit:
  - If accepted: $\text{Final} = y_{\text{cand}}$.
  - If rejected: $\text{Final} = T_0$.

---

## 4. Architectural Invariants & Data Governance

| Invariant | Mathematical Formulation | Failure Handling |
| :--- | :--- | :--- |
| **$K=1$ Single-Pass** | Exactly 1 intervention per aspect | Multi-aspect tweets factored into independent episodes |
| **Fail-Closed Firewall** | $\neg \text{PassFirewall} \implies \text{Final} = T_0$ | Bypasses $T_F$, reverts immediately to anchor |
| **Final Audit Safeguard** | $\text{AuditDecision} \neq \text{ACCEPT} \implies \text{Final} = T_0$ | Reverts candidate to anchor |
| **Zero Label Leakage** | $s_{\text{gold}} \notin \text{Trajectory}$ | `gold_pairs` strictly omitted from saved trajectory outputs |
| **Information-Flow Isolation** | $V_0 \not\to T_R$ | $V_0$ restricted to Controller; critique lexically sanitized |
| **Aspect Independence** | $\forall j \neq i, \quad a_j \text{ locked to } T_0(a_j)$ | Non-target aspects cannot drift during target evaluation |

---

## 5. MDP Transition Data Contract (`TrainingTransitionRecord`)

For subsequent Supervised Fine-Tuning (SFT) and Reinforcement Learning (PPO/GRPO/DPO), each transition record serialized in `trajectories.jsonl` contains the complete Markov state before the decision:

```json
{
  "aspect_index": 0,
  "decision_step": 0,
  "step": 1,
  "aspect_id": "tw15_dev_0001_Pedro_0",
  "span": [0, 5],
  "sample_id": "tw15_dev_0001",
  "aspect": "Pedro",
  "aspect_text": "Pedro",
  "t0_sentiment": "NEU",
  "pre_sentiment": "NEU",
  "state_before": {
    "anchor": {
      "aspect": "Pedro",
      "sentiment": "NEU",
      "reason": "...",
      "evidence": [...]
    },
    "visual_sketch": { "scene": "...", "objects": [...] },
    "budget": {
      "remaining_interventions": 1,
      "remaining_visual_probes": 1
    }
  },
  "route": "KEEP",
  "risk_type": "NO_RISK",
  "route_reason": "Text is clearly descriptive and factual.",
  "action": "KEEP",
  "action_mask": [1, 1, 1],
  "critique": null,
  "question": null,
  "raw_evidence": null,
  "verified_evidence": null,
  "candidate": null,
  "candidate_sentiment": null,
  "audit": null,
  "audit_decision": "N/A",
  "verifier_decision": "NO_OP",
  "post_sentiment": "NEU",
  "final_sentiment": "NEU",
  "compute": { ... }
}
```

---

## 6. Evaluation Protocol & Diagnostics

The evaluation system in `bacr/evaluator.py` computes two complementary diagnostics:

1. **Teacher Verification Hypothesis Diagnostic (`evaluate_v3_teacher`)**:
   - Compares $T_0$ accuracy vs Teacher Final accuracy.
   - Evaluates Marginal Utility:
     $$\Delta_{\text{Aspect}} = \text{Recover} - \text{Harm}$$
   - Measures route-level accuracy and intervention rates.

2. **Transition Governance Diagnostic (`evaluate_v3_trajectories`)**:
   - Aggregates action distributions (`KEEP`, `TEXT_RETHINK`, `VISION_PROBE`) from active `transitions`.
   - Tracks verifier decisions (`ACCEPT_REVISION`, `REVERT_TEXT_BASELINE`, `NO_OP`).
   - Evaluates system-level anchor corrections vs corruptions:
     $$\text{Net Gain} = \text{Corrections} - \text{Corruptions}$$
