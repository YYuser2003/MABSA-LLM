"""Benchmark Compilation Script for BACR Unified Evaluation Ladder.

Aggregates metrics across the standardized progression:
  G0 / T0 -> G1 Direct MM -> S0 Sketch -> G3 Vision Probe -> G4-TP Text Probe -> BACR

Outputs a clean GitHub-flavored Markdown table.
"""

import os
import sys
import json
import glob
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def get_latest_run_metrics(pattern: str) -> Optional[Dict[str, Any]]:
    matches = glob.glob(os.path.join(PROJECT_ROOT, "outputs", "runs", pattern, "metrics.json"))
    if not matches:
        return None
    # Sort by timestamp in directory name
    matches.sort()
    latest = matches[-1]
    return load_json(latest)


def format_row(method: str, m15: Optional[Dict[str, Any]], m17: Optional[Dict[str, Any]]) -> str:
    def extract_stats(m):
        if not m:
            return ("-", "-", "-", "-", "-")
        e2e = m.get("end_to_end_metrics", {})
        tsc = m.get("tsc_benchmark_metrics", {})
        comp = m.get("compute_efficiency", {})

        p_f1 = f"{e2e.get('pair_f1', 0.0):.2f}"
        acc = f"{tsc.get('accuracy_final', 0.0):.2f}"
        m_f1 = f"{tsc.get('macro_f1_final', 0.0):.2f}"
        calls = f"{comp.get('avg_api_calls', 0.0):.1f}"
        imgs = f"{comp.get('avg_image_invocations', 0.0):.1f}"
        return (p_f1, acc, m_f1, calls, imgs)

    f1_15, acc_15, mf1_15, calls_15, img_15 = extract_stats(m15)
    f1_17, acc_17, mf1_17, calls_17, img_17 = extract_stats(m17)

    return f"| {method:<20} | {f1_15:>7} | {acc_15:>7} | {mf1_15:>7} | {calls_15:>5} | {img_15:>4} | {f1_17:>7} | {acc_17:>7} | {mf1_17:>7} | {calls_17:>5} | {img_17:>4} |"


def main():
    print("=" * 110)
    print("                      MABSA-LLM UNIFIED BENCHMARK PROGRESSION REPORT")
    print("=" * 110)

    # 1. G0 / T0
    g0_15 = get_latest_run_metrics("*g0_t0*twitter2015_test*")
    g0_17 = get_latest_run_metrics("*g0_t0*twitter2017_test*")

    # 2. G1 Direct MM
    g1_15 = load_json(os.path.join(PROJECT_ROOT, "outputs", "archive", "gemini_direct_mm", "metrics_tw15_test.json"))
    g1_17 = load_json(os.path.join(PROJECT_ROOT, "outputs", "archive", "gemini_direct_mm", "metrics_tw17_test.json"))

    # 3. S0 Sketch
    s0_15 = get_latest_run_metrics("*s0_sketch*twitter2015_test*")
    s0_17 = get_latest_run_metrics("*s0_sketch*twitter2017_test*")

    # 4. G3 Vision Probe
    g3_15 = get_latest_run_metrics("*g3_vision_probe*twitter2015_test*")
    if not g3_15:
        g3_15 = load_json(os.path.join(PROJECT_ROOT, "outputs", "archive", "04_g3_precanonical", "metrics_g3_max2_twitter2015_test.json"))
    g3_17 = get_latest_run_metrics("*g3_vision_probe*twitter2017_test*")
    if not g3_17:
        g3_17 = load_json(os.path.join(PROJECT_ROOT, "outputs", "archive", "04_g3_precanonical", "metrics_g3_max2_twitter2017_test.json"))

    # 5. G4-TP Text Probe
    g4_15 = get_latest_run_metrics("*g4_tp*twitter2015_test*")
    g4_17 = get_latest_run_metrics("*g4_tp*twitter2017_test*")

    # 6. BACR
    bacr_15 = get_latest_run_metrics("*bacr*twitter2015_test*")
    bacr_17 = get_latest_run_metrics("*bacr*twitter2017_test*")

    header = (
        "| Method               | Twitter-2015 Test                                   | Twitter-2017 Test                                   |\n"
        "|                      | Pair F1 | Acc (%) | Mac F1  | Calls | Img  | Pair F1 | Acc (%) | Mac F1  | Calls | Img  |\n"
        "|:---------------------|:-------:|:-------:|:-------:|:-----:|:----:|:-------:|:-------:|:-------:|:-----:|:----:|"
    )

    rows = [
        format_row("G0 / T0 (Text-only)", g0_15, g0_17),
        format_row("G1 (Direct MM)", g1_15, g1_17),
        format_row("S0 (Global Sketch)", s0_15, s0_17),
        format_row("G3 (Vision Probe)", g3_15, g3_17),
        format_row("G4-TP (Text Probe)", g4_15, g4_17),
        format_row("BACR (Bi-Probe)", bacr_15, bacr_17),
    ]

    table = header + "\n" + "\n".join(rows)
    print("\n" + table + "\n")


if __name__ == "__main__":
    main()
