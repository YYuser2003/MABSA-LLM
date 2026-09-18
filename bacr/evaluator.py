"""Standalone Evaluator for Active Visual Reasoning Protocol (G3 v1.1).

Computes:
1. End-to-End Pair Precision, Recall, Micro-F1
2. Aspect Term Extraction (ATE) Precision, Recall, F1
3. Conditional Sentiment Accuracy (ASC)
4. Locked-Aspect Sentiment Metrics (isolating ASC gains from ATE recall):
   - SentAcc_locked
   - PairF1_locked
5. Aspect-Level Sentiment Diagnostics:
   - ERR_sent (Aspect Sentiment Error Recovery Rate: W -> C)
   - HRR_sent (Aspect Sentiment Harmful Revision Rate: C -> W)
   - VNG_sent (Aspect Net Sentiment Gain)
   - VINB (Visual Interaction Net Benefit: Net Recovery / Total Aspects)
6. Sample-Level Exact Match Transitions (C->C, W->C, C->W, W->W)
7. Stop Type Distribution: BFSR (Budget-Forced Stop Rate)
8. Test-Time Compute Efficiency:
   - PairF1 / API Call
   - PairF1 / 1K Tokens
   - PairF1 / Image Invocation
   - Avg Total Tokens & Latency
9. Query Taxonomy Distribution & Recovery Attribution
"""

import os
import sys
import json
import argparse
from collections import Counter
from typing import Dict, Any, List, Set, Tuple


def normalize_pair(pair: List[str]) -> Tuple[str, str]:
    """Normalizes aspect text and sentiment polarity."""
    aspect = str(pair[0]).strip().lower()
    sentiment = str(pair[1]).strip().upper()
    return (aspect, sentiment)


def evaluate_pair_set(
    pred_pairs_raw: List[List[str]],
    gold_pairs_raw: List[List[str]]
) -> Tuple[int, int, int]:
    """Computes TP, FP, FN for aspect-sentiment pairs."""
    pred_set = set(normalize_pair(p) for p in pred_pairs_raw)
    gold_set = set(normalize_pair(p) for p in gold_pairs_raw)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def evaluate_aspect_set(
    pred_pairs_raw: List[List[str]],
    gold_pairs_raw: List[List[str]]
) -> Tuple[int, int, int]:
    """Computes TP, FP, FN for aspect term extraction only."""
    pred_set = set(normalize_pair(p)[0] for p in pred_pairs_raw)
    gold_set = set(normalize_pair(p)[0] for p in gold_pairs_raw)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def compute_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Calculates precision, recall, and F1 score."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return round(p * 100, 2), round(r * 100, 2), round(f1 * 100, 2)


def extract_pairs(record: Dict[str, Any], key: str = "final_pairs") -> List[List[str]]:
    """Robustly extracts [aspect, sentiment] pairs from various JSON formats."""
    raw = record.get(key)
    if not raw and key == "final_pairs":
        raw = record.get("predictions", [])
    if isinstance(raw, list):
        pairs = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append([str(item[0]), str(item[1])])
            elif isinstance(item, dict) and "aspect" in item and "sentiment" in item:
                pairs.append([str(item["aspect"]), str(item["sentiment"])])
        return pairs
    return []


class G3Evaluator:
    def __init__(self, gold_file: str):
        self.gold_map: Dict[str, Dict[str, Any]] = {}
        with open(gold_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                sid = data.get("sample_id")
                if sid:
                    self.gold_map[sid] = data

    def evaluate_predictions(self, pred_file: str) -> Dict[str, Any]:
        """Runs full evaluation on G3 prediction JSONL."""
        preds: List[Dict[str, Any]] = []
        with open(pred_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                preds.append(json.loads(line))

        # Metrics accumulators
        tp_pair_final, fp_pair_final, fn_pair_final = 0, 0, 0
        tp_pair_init, fp_pair_init, fn_pair_init = 0, 0, 0
        tp_asp, fp_asp, fn_asp = 0, 0, 0

        # Locked-Aspect metrics accumulator
        tp_locked, fp_locked, fn_locked = 0, 0, 0

        cond_sent_correct, cond_sent_total = 0, 0
        total_gold_aspects = 0

        # Sample-Level Flow counters
        sample_cc, sample_wc, sample_cw, sample_ww = 0, 0, 0, 0

        # Aspect-Level Sentiment Flow counters
        asp_wc_count = 0  # Aspect: Initial Wrong -> Final Correct (Recovery)
        asp_cw_count = 0  # Aspect: Initial Correct -> Final Wrong (Harmful)
        asp_cc_count = 0  # Aspect: Initial Correct -> Final Correct
        asp_ww_count = 0  # Aspect: Initial Wrong -> Final Wrong

        total_queries = 0
        total_samples = 0
        stop_type_counts = {"natural_stop": 0, "budget_forced_stop": 0}
        taxonomy_counts: Dict[str, int] = {}
        taxonomy_recoveries: Dict[str, int] = {}

        # Compute tracking accumulators
        total_api_calls = 0
        total_tokens = 0
        total_image_invocations = 0
        total_latency_ms = 0.0
        has_compute_info = False

        # 3-Class Sentiment Tracking (for TSC benchmark parity)
        y_true_all: List[str] = []
        y_pred_final_all: List[str] = []
        y_pred_init_all: List[str] = []

        for p in preds:
            sid = p.get("sample_id")
            if sid not in self.gold_map:
                continue

            gold_item = self.gold_map[sid]
            gold_pairs = gold_item.get("pairs", [])
            total_gold_aspects += len(gold_pairs)

            # Predictions
            text_init = p.get("text_initial", {})
            if isinstance(text_init, dict):
                text_init_pairs = extract_pairs(text_init, "pairs")
                if not text_init_pairs and "aspects" in text_init:
                    text_init_pairs = [[a.get("text", a.get("aspect", "")), a.get("sentiment", "NEU")] for a in text_init["aspects"]]
            else:
                text_init_pairs = []
            if not text_init_pairs and "initial_pairs" in p:
                text_init_pairs = extract_pairs(p, "initial_pairs")

            final_pairs = extract_pairs(p, "final_pairs")
            if not final_pairs and "pairs" in p:
                final_pairs = extract_pairs(p, "pairs")
            if not text_init_pairs:
                text_init_pairs = final_pairs

            num_queries = p.get("num_visual_queries", 0) + p.get("num_text_queries", 0)
            stop_type = p.get("stop_type", "natural_stop")
            stop_type_counts[stop_type] = stop_type_counts.get(stop_type, 0) + 1

            total_samples += 1
            total_queries += num_queries

            # Taxonomy counts
            for r in p.get("rounds", []):
                qt = r.get("query_type", "GENERAL")
                taxonomy_counts[qt] = taxonomy_counts.get(qt, 0) + 1

            # Compute stats
            c_info = p.get("compute")
            if c_info:
                has_compute_info = True
                total_api_calls += c_info.get("api_calls_total", 0)
                total_tokens += c_info.get("total_tokens", 0)
                total_image_invocations += c_info.get("image_invocations", 0)
                total_latency_ms += c_info.get("latency_ms", 0.0)

            # 1. End-to-End Pair Evaluation
            tp, fp, fn = evaluate_pair_set(final_pairs, gold_pairs)
            tp_pair_final += tp
            fp_pair_final += fp
            fn_pair_final += fn

            # 2. Initial Text G0_R Pair Evaluation
            tpi, fpi, fni = evaluate_pair_set(text_init_pairs, gold_pairs)
            tp_pair_init += tpi
            fp_pair_init += fpi
            fn_pair_init += fni

            # 3. Aspect-Only Extraction Evaluation
            gold_asps = [x[0] for x in gold_pairs]
            pred_asps = [x[0] for x in final_pairs]
            tpa, fpa, fna = evaluate_aspect_set(pred_asps, gold_asps)
            tp_asp += tpa
            fp_asp += fpa
            fn_asp += fna

            # 4. Locked-Aspect Evaluation
            init_asp_set = set(normalize_pair(x)[0] for x in text_init_pairs)
            gold_locked_pairs = [x for x in gold_pairs if normalize_pair(x)[0] in init_asp_set]
            tpl, fpl, fnl = evaluate_pair_set(final_pairs, gold_locked_pairs)
            tp_locked += tpl
            fp_locked += fpl
            fn_locked += fnl

            # 5. Aspect-Level Sentiment Transitions & TSC Tracking
            gold_dict = dict(normalize_pair(x) for x in gold_pairs)
            init_dict = dict(normalize_pair(x) for x in text_init_pairs)
            final_dict = dict(normalize_pair(x) for x in final_pairs)

            for g_asp, g_sent in gold_dict.items():
                if g_asp in final_dict:
                    cond_sent_total += 1
                    if final_dict[g_asp] == g_sent:
                        cond_sent_correct += 1

                # TSC prediction alignment
                if g_asp in final_dict:
                    y_true_all.append(g_sent)
                    y_pred_final_all.append(final_dict[g_asp])
                    y_pred_init_all.append(init_dict.get(g_asp, "NEU"))

                # Matched aspects transitions
                if g_asp in init_dict and g_asp in final_dict:
                    init_sent_ok = (init_dict[g_asp] == g_sent)
                    final_sent_ok = (final_dict[g_asp] == g_sent)

                    if not init_sent_ok and final_sent_ok:
                        asp_wc_count += 1
                    elif init_sent_ok and not final_sent_ok:
                        asp_cw_count += 1
                    elif init_sent_ok and final_sent_ok:
                        asp_cc_count += 1
                    else:
                        asp_ww_count += 1

            # 6. Sample-Level Exact Match Transition
            init_set = set(normalize_pair(x) for x in text_init_pairs)
            final_set = set(normalize_pair(x) for x in final_pairs)
            gold_set = set(normalize_pair(x) for x in gold_pairs)

            if init_correct := (init_set == gold_set) and (final_set == gold_set):
                sample_cc += 1
            elif not init_correct and (final_set == gold_set):
                sample_wc += 1
                for r in p.get("rounds", []):
                    qt = r.get("query_type", "GENERAL")
                    taxonomy_recoveries[qt] = taxonomy_recoveries.get(qt, 0) + 1
            elif init_correct and not (final_set == gold_set):
                sample_cw += 1
            else:
                sample_ww += 1

        # Summary calculations
        p_final, r_final, f1_final = compute_f1(tp_pair_final, fp_pair_final, fn_pair_final)
        p_init, r_init, f1_init = compute_f1(tp_pair_init, fp_pair_init, fn_pair_init)
        p_asp, r_asp, f1_asp = compute_f1(tp_asp, fp_asp, fn_asp)
        p_lock, r_lock, f1_lock = compute_f1(tp_locked, fp_locked, fn_locked)

        cond_sent_acc = round((cond_sent_correct / cond_sent_total * 100), 2) if cond_sent_total > 0 else 0.0

        # TSC Accuracy & Macro-F1 across ["POS", "NEG", "NEU"]
        classes = ["POS", "NEG", "NEU"]
        f1_list_final = []
        f1_list_init = []
        class_metrics_final = {}
        class_metrics_init = {}

        total_tsc = len(y_true_all)
        if total_tsc > 0:
            corr_f = sum(1 for yt, yp in zip(y_true_all, y_pred_final_all) if yt == yp)
            acc_f = round((corr_f / total_tsc) * 100.0, 2)
            corr_i = sum(1 for yt, yp in zip(y_true_all, y_pred_init_all) if yt == yp)
            acc_i = round((corr_i / total_tsc) * 100.0, 2)

            for cls in classes:
                # Final
                tp_f = sum(1 for yt, yp in zip(y_true_all, y_pred_final_all) if yt == cls and yp == cls)
                fp_f = sum(1 for yt, yp in zip(y_true_all, y_pred_final_all) if yt != cls and yp == cls)
                fn_f = sum(1 for yt, yp in zip(y_true_all, y_pred_final_all) if yt == cls and yp != cls)
                prec_f = (tp_f / (tp_f + fp_f) * 100.0) if (tp_f + fp_f) > 0 else 0.0
                rec_f = (tp_f / (tp_f + fn_f) * 100.0) if (tp_f + fn_f) > 0 else 0.0
                f1_f = (2 * prec_f * rec_f / (prec_f + rec_f)) if (prec_f + rec_f) > 0 else 0.0
                f1_list_final.append(f1_f)
                class_metrics_final[cls] = {"precision": round(prec_f, 2), "recall": round(rec_f, 2), "f1-score": round(f1_f, 2)}

                # Init
                tp_i = sum(1 for yt, yp in zip(y_true_all, y_pred_init_all) if yt == cls and yp == cls)
                fp_i = sum(1 for yt, yp in zip(y_true_all, y_pred_init_all) if yt != cls and yp == cls)
                fn_i = sum(1 for yt, yp in zip(y_true_all, y_pred_init_all) if yt == cls and yp != cls)
                prec_i = (tp_i / (tp_i + fp_i) * 100.0) if (tp_i + fp_i) > 0 else 0.0
                rec_i = (tp_i / (tp_i + fn_i) * 100.0) if (tp_i + fn_i) > 0 else 0.0
                f1_i = (2 * prec_i * rec_i / (prec_i + rec_i)) if (prec_i + rec_i) > 0 else 0.0
                f1_list_init.append(f1_i)
                class_metrics_init[cls] = {"precision": round(prec_i, 2), "recall": round(rec_i, 2), "f1-score": round(f1_i, 2)}

            macro_f1_final = round(sum(f1_list_final) / len(f1_list_final), 2)
            macro_f1_init = round(sum(f1_list_init) / len(f1_list_init), 2)
        else:
            acc_f, acc_i, macro_f1_final, macro_f1_init = 0.0, 0.0, 0.0, 0.0

        anq = round(total_queries / total_samples, 2) if total_samples > 0 else 0.0
        gpq = round((f1_final - f1_init) / anq, 2) if anq > 0 else 0.0

        # BFSR (Budget Forced Stop Rate)
        bfsr = round(stop_type_counts.get("budget_forced_stop", 0) / total_samples * 100, 2) if total_samples > 0 else 0.0

        # Aspect-Level Sentiment Rates
        total_init_sent_errs = asp_wc_count + asp_ww_count
        total_init_sent_correct = asp_cc_count + asp_cw_count
        err_sent = round((asp_wc_count / total_init_sent_errs * 100), 2) if total_init_sent_errs > 0 else 0.0
        hrr_sent = round((asp_cw_count / total_init_sent_correct * 100), 2) if total_init_sent_correct > 0 else 0.0
        vng_sent = asp_wc_count - asp_cw_count

        # VINB (Visual Interaction Net Benefit)
        vinb = round((asp_wc_count - asp_cw_count) / total_gold_aspects * 100, 2) if total_gold_aspects > 0 else 0.0

        # Sample-Level Rates
        err_sample = round((sample_wc / (sample_wc + sample_ww) * 100), 2) if (sample_wc + sample_ww) > 0 else 0.0
        hrr_sample = round((sample_cw / (sample_cc + sample_cw) * 100), 2) if (sample_cc + sample_cw) > 0 else 0.0
        vng_sample = sample_wc - sample_cw

        # Compute efficiency calculations
        avg_api_calls = round(total_api_calls / total_samples, 2) if total_samples > 0 else 0.0
        avg_tokens = round(total_tokens / total_samples, 1) if total_samples > 0 else 0.0
        avg_img_invocations = round(total_image_invocations / total_samples, 2) if total_samples > 0 else 0.0
        avg_latency_s = round((total_latency_ms / total_samples) / 1000.0, 2) if total_samples > 0 else 0.0

        f1_per_api_call = round(f1_final / avg_api_calls, 2) if avg_api_calls > 0 else 0.0
        f1_per_1k_tokens = round(f1_final / (avg_tokens / 1000.0), 2) if avg_tokens > 0 else 0.0
        f1_per_image_invocation = round(f1_final / avg_img_invocations, 2) if avg_img_invocations > 0 else 0.0

        return {
            "total_samples": total_samples,
            "total_gold_aspects": total_gold_aspects,
            "end_to_end_metrics": {
                "pair_precision": p_final,
                "pair_recall": r_final,
                "pair_f1": f1_final,
                "aspect_f1": f1_asp,
                "cond_sent_acc": cond_sent_acc
            },
            "locked_aspect_metrics": {
                "pair_precision": p_lock,
                "pair_recall": r_lock,
                "pair_f1": f1_lock,
                "sent_acc_locked": cond_sent_acc
            },
            "initial_g0_r_metrics": {
                "pair_precision": p_init,
                "pair_recall": r_init,
                "pair_f1": f1_init
            },
            "delta_f1_over_g0_r": round(f1_final - f1_init, 2),
            "anq": anq,
            "gpq": gpq,
            "bfsr_percent": bfsr,
            "vinb_percent": vinb,
            "stop_type_distribution": stop_type_counts,
            "aspect_level_diagnostics": {
                "err_sent_percent": err_sent,
                "hrr_sent_percent": hrr_sent,
                "vng_sent_net_gain": vng_sent,
                "vinb_percent": vinb,
                "asp_wc_recovery": asp_wc_count,
                "asp_cw_harmful": asp_cw_count,
                "asp_cc_maintained": asp_cc_count,
                "asp_ww_unresolved": asp_ww_count
            },
            "sample_level_diagnostics": {
                "err_sample_percent": err_sample,
                "hrr_sample_percent": hrr_sample,
                "vng_sample_net_gain": vng_sample,
                "sample_wc_recovery": sample_wc,
                "sample_cw_harmful": sample_cw,
                "sample_cc_maintained": sample_cc,
                "sample_ww_unresolved": sample_ww
            },
            "compute_efficiency": {
                "has_compute_info": has_compute_info,
                "avg_api_calls": avg_api_calls,
                "avg_tokens": avg_tokens,
                "avg_image_invocations": avg_img_invocations,
                "avg_latency_s": avg_latency_s,
                "f1_per_api_call": f1_per_api_call,
                "f1_per_1k_tokens": f1_per_1k_tokens,
                "f1_per_image_invocation": f1_per_image_invocation
            },
            "tsc_benchmark_metrics": {
                "accuracy_final": acc_f,
                "macro_f1_final": macro_f1_final,
                "accuracy_init": acc_i,
                "macro_f1_init": macro_f1_init,
                "delta_accuracy": round(acc_f - acc_i, 2),
                "delta_macro_f1": round(macro_f1_final - macro_f1_init, 2),
                "class_metrics_final": class_metrics_final,
                "class_metrics_init": class_metrics_init
            },
            "taxonomy_distribution": taxonomy_counts,
            "taxonomy_recoveries": taxonomy_recoveries
        }


def print_evaluation_report(results: Dict[str, Any]):
    """Formats and prints the standard terminal evaluation report."""
    tsc = results.get("tsc_benchmark_metrics", {})
    em = results["end_to_end_metrics"]
    lm = results["locked_aspect_metrics"]
    im = results["initial_g0_r_metrics"]
    ad = results["aspect_level_diagnostics"]
    sd = results["sample_level_diagnostics"]
    st = results["stop_type_distribution"]
    cmp = results["compute_efficiency"]

    print("\n" + "=" * 78)
    print("      ACTIVE VISUAL REASONING PROTOCOL (G3 v1.2) EVALUATION REPORT")
    print("=" * 78)
    print(f"Total Evaluated Samples      : {results['total_samples']} (Total Gold Aspects: {results['total_gold_aspects']})")
    print("-" * 78)
    if tsc:
        print("1. TARGETED SENTIMENT CLASSIFICATION (Direct Paper Baseline Comparison):")
        print(f"  Final G3 Sentiment Accuracy : {tsc.get('accuracy_final')}%  | Macro-F1: {tsc.get('macro_f1_final')}%")
        print(f"  G0_R Text Sentiment Accuracy: {tsc.get('accuracy_init')}%  | Macro-F1: {tsc.get('macro_f1_init')}%")
        print(f"  Net Gain over Text (Δ)      : Acc: {tsc.get('delta_accuracy'):+0.2f}%, Macro-F1: {tsc.get('delta_macro_f1'):+0.2f}%")
        cm = tsc.get("class_metrics_final", {})
        print(f"  Class Breakdown (Final G3)  : POS F1={cm.get('POS', {}).get('f1-score', 0)}%, NEG F1={cm.get('NEG', {}).get('f1-score', 0)}%, NEU F1={cm.get('NEU', {}).get('f1-score', 0)}%")
        print("-" * 78)
    print("2. END-TO-END PAIR BENCHMARK METRICS (Standard Joint ATE+ASC Task):")
    print(f"  Final G3 Pair Micro-F1     : {em['pair_f1']}%  (P: {em['pair_precision']}%, R: {em['pair_recall']}%)")
    print(f"  G0_R Text Initial Pair F1  : {im['pair_f1']}%  (P: {im['pair_precision']}%, R: {im['pair_recall']}%)")
    print(f"  Net Pair F1 Gain (Δ)       : {results['delta_f1_over_g0_r']:+0.2f}%")
    print(f"  Aspect Extraction F1       : {em['aspect_f1']}% (Locked to text-init)")
    print(f"  Conditional Sentiment Acc  : {em['cond_sent_acc']}%")
    print("-" * 78)
    print("3. ACTIVE VISUAL REASONING BEHAVIORAL DIAGNOSTICS:")
    print(f"  Average Queries (ANQ)      : {results['anq']}  (Budget cap: max 2)")
    print(f"  Gain Per Query (GPQ)       : {results['gpq']}")
    print(f"  Budget Forced Stop Rate    : {results['bfsr_percent']}% (natural: {st.get('natural_stop', 0)}, forced: {st.get('budget_forced_stop', 0)})")
    print(f"  Visual Net Benefit (VINB)  : {results['vinb_percent']}%  ★ [NET ASPECT RESCUE]")
    print(f"  Aspect ERR_sent (Recovery) : {ad['err_sent_percent']}% ({ad['asp_wc_recovery']} aspects recovered)")
    print(f"  Aspect HRR_sent (Harmful)  : {ad['hrr_sent_percent']}% ({ad['asp_cw_harmful']} aspects damaged)")
    print(f"  Aspect VNG Net Gain        : {ad['vng_sent_net_gain']} aspects (ERR_sent >> HRR_sent)")
    print(f"  Sample-Level Exact Match   : CC={sd['sample_cc_maintained']}, WC={sd['sample_wc_recovery']}, CW={sd['sample_cw_harmful']}, WW={sd['sample_ww_unresolved']}")
    print("-" * 78)
    if cmp.get("has_compute_info"):
        print("4. TEST-TIME COMPUTE & RESOURCE EFFICIENCY ACCOUNTING:")
        print(f"  Avg API Calls / Sample     : {cmp['avg_api_calls']}")
        print(f"  Avg Total Tokens / Sample  : {cmp['avg_tokens']}")
        print(f"  Avg Image Invocations      : {cmp['avg_image_invocations']}")
        print(f"  Avg Latency / Sample       : {cmp['avg_latency_s']}s")
        print(f"  Efficiency (F1 / API Call) : {cmp['f1_per_api_call']}")
        print(f"  Efficiency (F1 / 1K Tokens): {cmp['f1_per_1k_tokens']}")
        print(f"  Efficiency (F1 / Image Call): {cmp['f1_per_image_invocation']}")
        print("-" * 78)
    print("5. QUERY TAXONOMY FREQUENCY & ATTRIBUTION:")
    for k, v in results.get("taxonomy_distribution", {}).items():
        rec = results.get("taxonomy_recoveries", {}).get(k, 0)
        print(f"  - {k:<15}: {v:>3} questions asked | {rec:>2} sample recoveries")
    print("=" * 78 + "\n")


BACREvaluator = G3Evaluator


def evaluate_v3_trajectories(traj_file: str, gold_file: str) -> Dict[str, Any]:
    """Computes transition-level BACR-v3 diagnostic metrics from trajectories and gold labels."""
    gold_map = {}
    gold_aspect_map = {}
    with open(gold_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                sid = d["sample_id"]
                gold_map[sid] = {tuple(p) for p in d.get("pairs", [])}
                gold_aspect_map[sid] = {p[0]: p[1] for p in d.get("pairs", [])}

    trajs = []
    with open(traj_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trajs.append(json.loads(line))

    total_samples = len(trajs)
    if total_samples == 0:
        return {"error": "Empty trajectory file"}

    # Accuracy accumulators
    ha_correct = 0
    hb0_correct = 0
    final_correct = 0

    # Marginal Utility accumulators
    recover_t, harm_t = 0, 0
    recover_v, harm_v = 0, 0

    # System-level anchor protection
    anchor_corrections = 0
    anchor_corruptions = 0

    action_counts = Counter()
    verifier_decisions = Counter()
    contrast_types = Counter()
    revert_count = 0

    for tr in trajs:
        sid = tr["sample_id"]
        gold_pairs = gold_map.get(sid, set())
        gold_asp_sents = gold_aspect_map.get(sid, {})

        # Predictions
        y_ha = {tuple(p) for p in tr.get("text_anchor", {}).get("pairs", tr.get("text_only", {}).get("pairs", []))}
        y_final = {tuple(p) for p in tr.get("final_pairs", tr.get("text_baseline", {}).get("pairs", []))}

        # Baseline at step 0
        y_hb0 = y_ha  # In BACR-v3, H_B(0) == H_A

        is_ha_correct = (y_ha == gold_pairs)
        is_hb0_correct = (y_hb0 == gold_pairs)
        is_final_correct = (y_final == gold_pairs)

        if is_ha_correct:
            ha_correct += 1
        if is_hb0_correct:
            hb0_correct += 1
        if is_final_correct:
            final_correct += 1

        # Anchor protection net gain
        if not is_ha_correct and is_final_correct:
            anchor_corrections += 1
        elif is_ha_correct and not is_final_correct:
            anchor_corruptions += 1

        # Controller initial action & contrast
        act = tr.get("action", "FINALIZE")
        action_counts[act] += 1
        c_type = tr.get("contrast_type", "NO_RISK")
        contrast_types[c_type] += 1

        # Transition-level evaluation
        transitions = tr.get("transitions", [])
        if not transitions and "rounds" in tr:
            # Reconstruct transitions from rounds if older format
            for rnd in tr["rounds"]:
                vdec = rnd.get("verifier_decision", {}).get("decision") or rnd.get("ct_decision", {}).get("decision", "REVERT_TEXT_BASELINE")
                if vdec in ["REVERT_TEXT_BASELINE", "REVERT_ANCHOR", "REJECT_REVISION"]:
                    revert_count += 1
                verifier_decisions[vdec] += 1
        else:
            for t in transitions:
                action = t.get("action")
                vdec = t.get("verifier_decision", "REVERT_TEXT_BASELINE")
                verifier_decisions[vdec] += 1
                if vdec in ["REVERT_TEXT_BASELINE", "REVERT_ANCHOR"]:
                    revert_count += 1

                # Aspect-level marginal utility calculation
                aid = t.get("aspect_id")
                pre = t.get("pre_sentiment")
                post = t.get("post_sentiment")
                gold_sent = gold_asp_sents.get(aid)

                if gold_sent:
                    was_correct = (pre == gold_sent)
                    now_correct = (post == gold_sent)

                    if action == "TEXT_RETHINK":
                        if not was_correct and now_correct:
                            recover_t += 1
                        elif was_correct and not now_correct:
                            harm_t += 1
                    elif action == "VISION_PROBE":
                        if not was_correct and now_correct:
                            recover_v += 1
                        elif was_correct and not now_correct:
                            harm_v += 1

    acc_ha = round(ha_correct / total_samples * 100, 2)
    acc_hb0 = round(hb0_correct / total_samples * 100, 2)
    acc_final = round(final_correct / total_samples * 100, 2)

    mu_t = recover_t - harm_t
    mu_v = recover_v - harm_v

    return {
        "total_samples": total_samples,
        "acc_ha": acc_ha,
        "acc_hb0": acc_hb0,
        "acc_final": acc_final,
        "acc_t0": acc_ha,  # Backward compatibility key
        "delta_final_over_ha": round(acc_final - acc_ha, 2),
        "marginal_utility_text": {
            "recover": recover_t,
            "harm": harm_t,
            "mu_t": mu_t
        },
        "marginal_utility_vision": {
            "recover": recover_v,
            "harm": harm_v,
            "mu_v": mu_v
        },
        "anchor_corrections": anchor_corrections,
        "anchor_corruptions": anchor_corruptions,
        "anchor_net_gain": anchor_corrections - anchor_corruptions,
        "action_distribution": dict(action_counts),
        "verifier_decisions": dict(verifier_decisions),
        "revert_count": revert_count,
        "contrast_distribution": dict(contrast_types)
    }


def print_v3_evaluation_report(r: Dict[str, Any]):
    print("\n" + "=" * 78)
    print("      BACR-v3 TRANSITION MARGINAL UTILITY & GOVERNANCE REPORT")
    print("=" * 78)
    print(f"Total Evaluated Samples         : {r.get('total_samples')}")
    print("-" * 78)
    print("1. PROGRESSIVE BASELINE ACCURACY:")
    print(f"  Text Anchor Baseline (H_A)      : {r.get('acc_ha')}%")
    print(f"  Verified Baseline (H_B[0])      : {r.get('acc_hb0')}%")
    print(f"  Final System Accuracy (Y_final) : {r.get('acc_final')}% (Δ: {r.get('delta_final_over_ha'):+0.2f}%)")
    print("-" * 78)
    print("2. MARGINAL UTILITIES (PRE -> POST TRANSITIONS):")
    mu_t = r.get("marginal_utility_text", {})
    print(f"  Text Rethink (MU_T)             : {mu_t.get('mu_t', 0):+d} (Recover: {mu_t.get('recover', 0)}, Harm: {mu_t.get('harm', 0)})")
    mu_v = r.get("marginal_utility_vision", {})
    print(f"  Vision Probe (MU_V)             : {mu_v.get('mu_v', 0):+d} (Recover: {mu_v.get('recover', 0)}, Harm: {mu_v.get('harm', 0)})")
    print("-" * 78)
    print("3. ANCHOR PROTECTION & SYSTEM NET GAIN:")
    print(f"  Anchor Corrections (H_A wrong -> Y_final correct): {r.get('anchor_corrections')}")
    print(f"  Anchor Corruptions (H_A correct -> Y_final wrong): {r.get('anchor_corruptions')}")
    print(f"  System Net Gain (Corrections - Corruptions)     : {r.get('anchor_net_gain'):+d}")
    print("-" * 78)
    print("4. CONTROLLER ACTIONS & VERIFIER AUDITS:")
    print(f"  Controller Actions Allocated    : {r.get('action_distribution')}")
    print(f"  Verifier Decisions              : {r.get('verifier_decisions')}")
    print(f"  Safeguard Reversions (REVERT)   : {r.get('revert_count')} times")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate G3 / BACR predictions.")
    parser.add_argument("--pred-file", required=True, help="Path to predictions JSONL file.")
    parser.add_argument("--gold-file", required=True, help="Path to gold dataset JSONL file.")
    parser.add_argument("--traj-file", default=None, help="Path to trajectories JSONL file (optional for v3 diagnostics).")
    args = parser.parse_args()

    evaluator = G3Evaluator(gold_file=args.gold_file)
    res = evaluator.evaluate_predictions(pred_file=args.pred_file)
    print_evaluation_report(res)

    if args.traj_file and os.path.exists(args.traj_file):
        v3_res = evaluate_v3_trajectories(args.traj_file, args.gold_file)
        print_v3_evaluation_report(v3_res)

