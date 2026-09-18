"""BACR-v3 Minimal Teacher Pipeline Runner.

Core Architecture:
    T0 + V0 + a_gold -> C_R -> {KEEP, TEXT, VISION} -> Candidate -> C_F -> Final

Invariants:
    1. K = 1: Single-pass intervention per gold aspect. Multi-aspect samples are
       factored into independent single-aspect evaluations and re-aggregated.
    2. EvidenceInvalid => T0: Fail-closed Evidence Firewall (C_E).
    3. AuditReject => T0: Unified Final Audit Verifier (C_F).
"""

import os
import time
import copy
from collections import Counter
from typing import Dict, Any, Optional, Union, List, Tuple, Set

from bacr.client import BaseClient
from bacr.meta_controller import MetaController
from bacr.text_reasoner import TextReasoner
from bacr.vision_sensor import VisionSensor
from bacr.schemas_v3 import (
    RouteAction,
    EvidenceStatus,
    TargetBinding,
    EvidenceRelevance,
    RevisionSupport,
    AuditDecision,
    CandidatePrediction,
    FinalAudit,
    RouteDecision,
    TrainingTransitionRecord
)


class BACRPipelineV3:
    """Minimal Teacher Pipeline implementing single-pass target-guided verification (K=1)."""

    def __init__(
        self,
        client: Optional[BaseClient] = None,
        run_id: Optional[str] = None,
        max_visual_probes: int = 1,
        k_interventions: int = 1,
        text_client: Optional[BaseClient] = None,
        controller_client: Optional[BaseClient] = None,
        vision_client: Optional[BaseClient] = None,
        **kwargs
    ):
        self.client = client
        self.run_id = run_id or f"bacr_v3_{int(time.time())}"
        self.max_visual_probes = max_visual_probes
        self.k_interventions = k_interventions

        if self.k_interventions != 1:
            raise ValueError(
                f"BACR-v3 Minimal Teacher currently strictly supports only K=1 (got k_interventions={self.k_interventions}). "
                "For multi-turn RL environment, use the dedicated BACREnv."
            )
        if self.max_visual_probes != 1:
            raise ValueError(
                f"BACR-v3 Minimal Teacher currently strictly supports only max_visual_probes=1 (got max_visual_probes={self.max_visual_probes})."
            )

        # Initialize sub-components
        self.text_reasoner = TextReasoner(client=text_client or client)
        self.controller = MetaController(client=controller_client or client)
        self.vision_sensor = VisionSensor(client=vision_client or client)

        # Compatibility alias
        self.meta_controller = self.controller

    def run_aspect(
        self,
        text: str,
        image_path: str,
        aspect: str,
        t0: Dict[str, Any],
        v0: Dict[str, Any],
        aspect_id: str = "",
        aspect_index: int = 0,
        span: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """Runs the minimal Teacher loop for a single gold aspect (K=1).
        
        Route:
          KEEP:   Final = T0
          TEXT:   TR -> CF -> Final
          VISION: IP -> CE (fail-closed) -> TF -> CF -> Final
        """
        compute = {
            "api_calls_text": 0,
            "api_calls_vision": 0,
            "api_calls_controller": 0,
            "api_calls_total": 0,
            "total_tokens": 0,
            "image_invocations": 0,
            "latency_ms": 0.0
        }

        anchor_sent = str(t0.get("sentiment", "NEU")).upper().strip()
        anchor_dict = {
            "aspect": aspect,
            "sentiment": anchor_sent,
            "reason": t0.get("reason", t0.get("rationale", "")),
            "evidence": t0.get("evidence", t0.get("text_evidence", []))
        }
        state_before = {
            "anchor": anchor_dict,
            "visual_sketch": v0,
            "budget": {
                "remaining_interventions": 1,
                "remaining_visual_probes": 1
            }
        }

        # Step 1: Controller Route Decision
        route_dec, usage_c, lat_c = self.controller.decide_route(
            aspect=aspect,
            anchor=anchor_dict,
            visual_sketch=v0
        )
        compute["api_calls_controller"] += 1
        compute["api_calls_total"] += 1
        compute["total_tokens"] += usage_c.get("total_tokens", 0)
        compute["latency_ms"] += lat_c * 1000

        action = route_dec.action
        candidate_dict: Optional[Dict[str, Any]] = None
        raw_evidence_dict: Optional[Dict[str, Any]] = None
        evidence_dict: Optional[Dict[str, Any]] = None
        audit_dict: Optional[Dict[str, Any]] = None
        final_sentiment = anchor_sent
        critique_text: Optional[str] = None

        # --------------------------------------------------------------------
        # Route 1: KEEP
        # --------------------------------------------------------------------
        if action == RouteAction.KEEP:
            final_sentiment = anchor_sent

        # --------------------------------------------------------------------
        # Route 2: TEXT Re-deliberation
        # --------------------------------------------------------------------
        elif action == RouteAction.TEXT:
            critique_text = route_dec.critique or self.controller.generate_text_critique(
                aspect=aspect,
                anchor=anchor_dict,
                risk_type=route_dec.risk_type
            )
            cand, usage_t, lat_t = self.text_reasoner.rethink_text(
                text=text,
                aspect=aspect,
                anchor=anchor_dict,
                critique=critique_text
            )
            compute["api_calls_text"] += 1
            compute["api_calls_total"] += 1
            compute["total_tokens"] += usage_t.get("total_tokens", 0)
            compute["latency_ms"] += lat_t * 1000
            candidate_dict = cand.model_dump()

            audit, usage_a, lat_a = self.controller.audit_revision(
                route="TEXT",
                text=text,
                aspect=aspect,
                anchor=anchor_dict,
                candidate=candidate_dict
            )
            compute["api_calls_controller"] += 1
            compute["api_calls_total"] += 1
            compute["total_tokens"] += usage_a.get("total_tokens", 0)
            compute["latency_ms"] += lat_a * 1000
            audit_dict = audit.model_dump()

            if audit.decision == AuditDecision.ACCEPT:
                final_sentiment = cand.sentiment
            else:
                final_sentiment = anchor_sent

        # --------------------------------------------------------------------
        # Route 3: VISION Probe + Evidence Firewall
        # --------------------------------------------------------------------
        elif action == RouteAction.VISION:
            if not route_dec.question:
                # Double safety guard: cannot probe vision without an explicit factual question
                final_sentiment = anchor_sent
                audit_dict = {
                    "decision": AuditDecision.REVERT.value,
                    "reason": "Firewall fail-closed: VISION route had no valid question; safely defaulted to T0 baseline."
                }
            else:
                question = route_dec.question
                raw_obs, usage_v, lat_v = self.vision_sensor.probe_deep(
                    image_path=image_path,
                    question=question
                )
                raw_evidence_dict = raw_obs
                compute["api_calls_vision"] += 1
                compute["api_calls_total"] += 1
                compute["image_invocations"] += 1
                compute["total_tokens"] += usage_v.get("total_tokens", 0)
                compute["latency_ms"] += lat_v * 1000

                ev_res, usage_ce, lat_ce = self.controller.filter_evidence(
                    aspect=aspect,
                    question=question,
                    raw_evidence=raw_obs
                )
                compute["api_calls_controller"] += 1
                compute["api_calls_total"] += 1
                compute["total_tokens"] += usage_ce.get("total_tokens", 0)
                compute["latency_ms"] += lat_ce * 1000
                evidence_dict = ev_res.model_dump()

                # Fail-closed Firewall Check: VALID + DIRECT + HIGH + SUPPORTS_REVISION
                is_valid_evidence = (
                    ev_res.status == EvidenceStatus.VALID
                    and ev_res.target_binding == TargetBinding.DIRECT
                    and ev_res.relevance == EvidenceRelevance.HIGH
                    and ev_res.revision_support == RevisionSupport.SUPPORTS_REVISION
                    and len(ev_res.usable_evidence) > 0
                )
                if not is_valid_evidence:
                    final_sentiment = anchor_sent
                    audit_dict = {
                        "decision": AuditDecision.REVERT.value,
                        "reason": f"Firewall fail-closed: status={ev_res.status.value}, binding={ev_res.target_binding.value}, relevance={ev_res.relevance.value}, support={ev_res.revision_support.value}."
                    }
                else:
                    cand, usage_tf, lat_tf = self.text_reasoner.fuse_evidence(
                        text=text,
                        aspect=aspect,
                        anchor=anchor_dict,
                        verified_evidence=evidence_dict
                    )
                    compute["api_calls_text"] += 1
                    compute["api_calls_total"] += 1
                    compute["total_tokens"] += usage_tf.get("total_tokens", 0)
                    compute["latency_ms"] += lat_tf * 1000
                    candidate_dict = cand.model_dump()

                    audit, usage_a, lat_a = self.controller.audit_revision(
                        route="VISION",
                        text=text,
                        aspect=aspect,
                        anchor=anchor_dict,
                        candidate=candidate_dict,
                        evidence=evidence_dict
                    )
                    compute["api_calls_controller"] += 1
                    compute["api_calls_total"] += 1
                    compute["total_tokens"] += usage_a.get("total_tokens", 0)
                    compute["latency_ms"] += lat_a * 1000
                    audit_dict = audit.model_dump()

                    if audit.decision == AuditDecision.ACCEPT:
                        final_sentiment = cand.sentiment
                    else:
                        final_sentiment = anchor_sent

        return {
            "aspect": aspect,
            "aspect_id": aspect_id or f"{aspect}_{aspect_index}",
            "aspect_index": aspect_index,
            "span": span,
            "state_before": state_before,
            "t0_sentiment": anchor_sent,
            "route": action.value,
            "risk_type": route_dec.risk_type,
            "route_reason": route_dec.reason,
            "critique": critique_text if action == RouteAction.TEXT else None,
            "question": route_dec.question if action == RouteAction.VISION else None,
            "candidate": candidate_dict,
            "candidate_sentiment": candidate_dict.get("sentiment") if candidate_dict else None,
            "raw_evidence": raw_evidence_dict,
            "verified_evidence": evidence_dict,
            "evidence": evidence_dict,
            "audit": audit_dict,
            "audit_decision": audit_dict.get("decision") if audit_dict else "N/A",
            "final_sentiment": final_sentiment,
            "compute": compute
        }

    def run_sample(
        self,
        sample: Dict[str, Any],
        image_base_dir: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """Executes BACR-v3 Minimal Teacher inference on a sample.
        Iterates gold aspects independently and re-aggregates final pairs.
        """
        sid = sample.get("sample_id", "unknown")
        text = sample.get("text", "")
        raw_image = sample.get("image", "")

        # Resolve image path
        image_path = raw_image
        if image_base_dir and not os.path.isabs(raw_image):
            cand_paths = [
                os.path.join(image_base_dir, raw_image),
                os.path.join(image_base_dir, "data", "twitter2015_images", raw_image),
                os.path.join(image_base_dir, "data", "twitter2017_images", raw_image),
                os.path.join(image_base_dir, "data", raw_image)
            ]
            for cp in cand_paths:
                if os.path.exists(cp):
                    image_path = cp
                    break

        # Resolve gold aspects
        gold_pairs = sample.get("pairs", [])
        if gold_pairs and isinstance(gold_pairs[0], dict):
            gold_pairs = [[p.get("aspect", p.get("text", "")), p.get("sentiment", "NEU")] for p in gold_pairs]

        # Resolve canonical requirements
        require_canonical_t0 = sample.get("require_canonical_t0", kwargs.get("require_canonical_t0", False))
        require_canonical_v0 = sample.get("require_canonical_v0", kwargs.get("require_canonical_v0", False))

        # Resolve T0 cache
        t0_cache = sample.get("text_initial_cached", sample.get("t0_cached"))
        if require_canonical_t0 and not t0_cache:
            raise RuntimeError(
                f"Sample '{sid}' requires canonical T0 cache (canonical_t0: true), but text_initial_cached is missing!"
            )

        cached_aspect_list: List[Dict[str, Any]] = []
        if t0_cache and isinstance(t0_cache, dict):
            raw_aspects = t0_cache.get("aspects") or t0_cache.get("text_initial", {}).get("aspects", [])
            if raw_aspects and isinstance(raw_aspects, list):
                for c_idx, a in enumerate(raw_aspects):
                    c_text = str(a.get("text", a.get("aspect", "")))
                    c_span = a.get("span")
                    if isinstance(c_span, (list, tuple)) and len(c_span) == 2:
                        c_span = tuple(c_span)
                    else:
                        c_span = None
                    c_aid = str(a.get("aspect_id", ""))
                    c_sent = str(a.get("sentiment", "NEU")).upper().strip()
                    c_reason = str(a.get("reason", a.get("rationale", "Cached canonical T0")))
                    cached_aspect_list.append({
                        "text": c_text,
                        "aspect_id": c_aid,
                        "span": c_span,
                        "sentiment": c_sent,
                        "reason": c_reason,
                        "index": c_idx
                    })
            else:
                raw_pairs = t0_cache.get("pairs") or t0_cache.get("text_initial", {}).get("pairs", [])
                if raw_pairs and isinstance(raw_pairs, list):
                    for c_idx, p in enumerate(raw_pairs):
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            p_text = str(p[0])
                            p_sent = str(p[1]).upper().strip()
                        elif isinstance(p, dict):
                            p_text = str(p.get("aspect", p.get("text", "")))
                            p_sent = str(p.get("sentiment", "NEU")).upper().strip()
                        else:
                            p_text = str(p)
                            p_sent = "NEU"
                        cached_aspect_list.append({
                            "text": p_text,
                            "aspect_id": "",
                            "span": None,
                            "sentiment": p_sent,
                            "reason": "Cached canonical T0",
                            "index": c_idx
                        })

        # Target aspects to evaluate (preserving aspect index and spans)
        target_aspect_items: List[Dict[str, Any]] = []
        annotations = sample.get("annotations", [])
        if annotations and isinstance(annotations, list) and len(annotations) > 0:
            for idx, ann in enumerate(annotations):
                asp_name = ann.get("aspect", "")
                raw_span = ann.get("span")
                span = tuple(raw_span) if isinstance(raw_span, (list, tuple)) and len(raw_span) == 2 else None
                aid = str(ann.get("aspect_id") or f"{sid}_{asp_name}_{idx}")
                target_aspect_items.append({
                    "aspect": asp_name,
                    "aspect_index": idx,
                    "aspect_id": aid,
                    "span": span
                })
        elif gold_pairs:
            for idx, p in enumerate(gold_pairs):
                asp_name = p[0] if isinstance(p, (list, tuple)) else (p.get("aspect", "") if isinstance(p, dict) else str(p))
                target_aspect_items.append({
                    "aspect": asp_name,
                    "aspect_index": idx,
                    "aspect_id": f"{sid}_{asp_name}_{idx}",
                    "span": None
                })
        elif cached_aspect_list:
            for idx, c in enumerate(cached_aspect_list):
                target_aspect_items.append({
                    "aspect": c["text"],
                    "aspect_index": idx,
                    "aspect_id": c["aspect_id"] or f"{sid}_{c['text']}_{idx}",
                    "span": c["span"]
                })
        else:
            target_aspect_items.append({
                "aspect": "Target Aspect",
                "aspect_index": 0,
                "aspect_id": f"{sid}_Target Aspect_0",
                "span": None
            })

        # Precompute term occurrences to accurately align duplicate aspect occurrences
        target_occ_counter: Dict[str, int] = Counter()
        target_item_occ: Dict[int, int] = {}
        for item in target_aspect_items:
            k_term = item["aspect"].strip().lower()
            target_item_occ[item["aspect_index"]] = target_occ_counter[k_term]
            target_occ_counter[k_term] += 1

        cache_occ_counter: Dict[str, int] = Counter()
        cache_occ_to_idx: Dict[Tuple[str, int], int] = {}
        for c_idx, c in enumerate(cached_aspect_list):
            k_term = c["text"].strip().lower()
            cache_occ_to_idx[(k_term, cache_occ_counter[k_term])] = c_idx
            cache_occ_counter[k_term] += 1

        # Resolve V0 cache or perceive globally
        v0_cache = sample.get("image_initial_cached", sample.get("v0_cached"))
        if require_canonical_v0 and (not v0_cache or not isinstance(v0_cache, dict)):
            raise RuntimeError(
                f"Sample '{sid}' requires canonical V0 cache (canonical_v0: true), but image_initial_cached is missing or invalid!"
            )

        total_compute = {
            "api_calls_text": 0,
            "api_calls_vision": 0,
            "api_calls_controller": 0,
            "api_calls_total": 0,
            "total_tokens": 0,
            "image_invocations": 0,
            "latency_ms": 0.0
        }

        if v0_cache and isinstance(v0_cache, dict):
            v0 = v0_cache
        else:
            v0, v0_usage, v0_lat = self.vision_sensor.perceive_global(image_path=image_path)
            total_compute["api_calls_vision"] += 1
            total_compute["api_calls_total"] += 1
            total_compute["image_invocations"] += 1
            total_compute["total_tokens"] += v0_usage.get("total_tokens", 0)
            total_compute["latency_ms"] += v0_lat * 1000

        # Execute single-aspect Teacher reasoning
        aspect_trajectories: List[Dict[str, Any]] = []
        final_pairs: List[List[str]] = []
        transitions: List[Dict[str, Any]] = []
        t0_pairs: List[List[str]] = []

        used_cache_indices: Set[int] = set()

        for step_idx, asp_item in enumerate(target_aspect_items):
            asp = asp_item["aspect"]
            asp_idx = asp_item["aspect_index"]
            aid = asp_item["aspect_id"]
            span = asp_item["span"]
            k = asp.strip().lower()
            term_occ = target_item_occ.get(asp_idx, 0)

            # Match T0 item from cached_aspect_list
            matched_c_idx = None

            # Priority 1: Match by exact span and aspect text
            if span is not None:
                for c_idx, c in enumerate(cached_aspect_list):
                    if c_idx not in used_cache_indices and c["span"] == span and c["text"].strip().lower() == k:
                        matched_c_idx = c_idx
                        break

            # Priority 2: Match by exact aspect_id and aspect text
            if matched_c_idx is None and aid:
                for c_idx, c in enumerate(cached_aspect_list):
                    if c_idx not in used_cache_indices and c["aspect_id"] and c["aspect_id"] == aid and c["text"].strip().lower() == k:
                        matched_c_idx = c_idx
                        break

            # Priority 3: Match by identical occurrence count of this term
            if matched_c_idx is None:
                cand_idx = cache_occ_to_idx.get((k, term_occ))
                if cand_idx is not None and cand_idx not in used_cache_indices:
                    matched_c_idx = cand_idx

            # Priority 4: Match by positional index if text matches
            if matched_c_idx is None and asp_idx < len(cached_aspect_list) and asp_idx not in used_cache_indices:
                if cached_aspect_list[asp_idx]["text"].strip().lower() == k:
                    matched_c_idx = asp_idx

            # Priority 5: Match any remaining unused occurrence with matching text
            if matched_c_idx is None:
                for c_idx, c in enumerate(cached_aspect_list):
                    if c_idx not in used_cache_indices and c["text"].strip().lower() == k:
                        matched_c_idx = c_idx
                        break

            if matched_c_idx is not None:
                used_cache_indices.add(matched_c_idx)
                c_hit = cached_aspect_list[matched_c_idx]
                t0_item = {
                    "text": c_hit["text"],
                    "aspect_id": c_hit["aspect_id"] or aid,
                    "span": c_hit["span"] or span,
                    "sentiment": c_hit["sentiment"],
                    "reason": c_hit["reason"]
                }
            else:
                if require_canonical_t0:
                    raise RuntimeError(
                        f"Sample '{sid}' aspect '{asp}' (index {asp_idx}, span {span}, id '{aid}') "
                        f"could not be resolved from canonical T0 cache! Available cached aspects: "
                        f"{[{'text': c['text'], 'span': c['span'], 'id': c['aspect_id']} for c in cached_aspect_list]}."
                    )
                # Fallback to default NEU baseline (exact match only, zero substring bleeding)
                t0_item = {"text": asp, "sentiment": "NEU", "reason": "Default text baseline"}

            t0_sent = t0_item.get("sentiment", "NEU")
            t0_pairs.append([asp, t0_sent])

            asp_res = self.run_aspect(
                text=text,
                image_path=image_path,
                aspect=asp,
                t0=t0_item,
                v0=v0,
                aspect_id=aid,
                aspect_index=asp_idx,
                span=span
            )
            aspect_trajectories.append(asp_res)
            final_pairs.append([asp, asp_res["final_sentiment"]])

            # Merge compute
            for comp_k, comp_v in asp_res["compute"].items():
                if comp_k in total_compute:
                    total_compute[comp_k] += comp_v

            # Build rich training-compatible transition record validated by TrainingTransitionRecord schema
            action_name = "TEXT_RETHINK" if asp_res["route"] == RouteAction.TEXT.value else ("VISION_PROBE" if asp_res["route"] == RouteAction.VISION.value else "KEEP")
            vdec = "ACCEPT_REVISION" if asp_res["audit_decision"] == AuditDecision.ACCEPT.value else ("REVERT_TEXT_BASELINE" if asp_res["route"] != RouteAction.KEEP.value else "NO_OP")
            trans_obj = TrainingTransitionRecord(
                aspect_index=asp_idx,
                decision_step=0,
                step=step_idx + 1,
                aspect_id=aid,
                span=span,
                sample_id=sid,
                aspect=asp,
                aspect_text=asp,
                t0_sentiment=t0_sent,
                pre_sentiment=t0_sent,
                state_before=asp_res.get("state_before", {}),
                route=asp_res["route"],
                risk_type=asp_res.get("risk_type", "NO_RISK"),
                route_reason=asp_res.get("route_reason", ""),
                action=action_name,
                action_mask=[1, 1, 1],
                critique=asp_res.get("critique"),
                question=asp_res.get("question"),
                raw_evidence=asp_res.get("raw_evidence"),
                verified_evidence=asp_res.get("verified_evidence"),
                evidence=asp_res.get("evidence"),
                candidate=asp_res.get("candidate"),
                candidate_sentiment=asp_res.get("candidate_sentiment"),
                audit=asp_res.get("audit"),
                audit_decision=asp_res.get("audit_decision", "N/A"),
                verifier_decision=vdec,
                post_sentiment=asp_res["final_sentiment"],
                final_sentiment=asp_res["final_sentiment"],
                compute=asp_res["compute"]
            )
            # Record ALL aspect decisions including KEEP to prevent selection bias in downstream SFT/RL
            transitions.append(trans_obj.model_dump())

        num_v = sum(1 for at in aspect_trajectories if at["route"] == RouteAction.VISION.value)
        num_t = sum(1 for at in aspect_trajectories if at["route"] == RouteAction.TEXT.value)

        return {
            "sample_id": sid,
            "text": text,
            "image": raw_image,
            "t0_pairs": t0_pairs,
            "final_pairs": final_pairs,
            "text_anchor": {"pairs": t0_pairs},
            "text_baseline": {"pairs": t0_pairs},
            "contrast_type": aspect_trajectories[0]["risk_type"] if aspect_trajectories else "NO_RISK",
            "aspect_trajectories": aspect_trajectories,
            "transitions": transitions,
            "num_visual_probes": num_v,
            "num_text_queries": num_t,
            "num_queries_total": num_v + num_t,
            "compute": total_compute,
            "stop_type": "natural_stop"
        }

