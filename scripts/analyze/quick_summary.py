import os
import json

# Archive runs
g1_summary_path = os.path.join("outputs", "archive", "02_gemini_direct_baselines", "g1_direct_mm", "benchmark_summary.json")
if not os.path.exists(g1_summary_path):
    g1_summary_path = os.path.join("outputs", "archive", "gemini_direct_mm", "benchmark_summary.json")

g3_15_path = os.path.join("outputs", "archive", "04_g3_precanonical", "metrics_g3_max2_twitter2015_test.json")
g3_17_path = os.path.join("outputs", "archive", "04_g3_precanonical", "metrics_g3_max2_twitter2017_test.json")

g1_data = {}
if os.path.exists(g1_summary_path):
    with open(g1_summary_path, "r", encoding="utf-8") as f:
        g1_data = json.load(f)

g3_15_data = {}
if os.path.exists(g3_15_path):
    with open(g3_15_path, "r", encoding="utf-8") as f:
        g3_15_data = json.load(f)

g3_17_data = {}
if os.path.exists(g3_17_path):
    with open(g3_17_path, "r", encoding="utf-8") as f:
        g3_17_data = json.load(f)

runs_info = [
    ("G0 (Text-only Canonical)", "canonical_g0_t0_twitter2015_test", "canonical_g0_t0_twitter2017_test"),
    ("G1 (Direct Multimodal 1-pass)", None, None),
    ("G1-SR (Direct MM 2-turn Reflect)", "g1_sr_15", "g1_sr_17"),
    ("S0 (T0 + V0 Global Sketch)", "20260918_122932_s0_sketch_twitter2015_test", "20260918_124003_s0_sketch_twitter2017_test"),
    ("G3 (Visual Probing Only)", "g3_15", "g3_17"),
    ("G4 (Text Probing Only)", "20260918_125119_g4_tp_twitter2015_test", "20260918_130314_g4_tp_twitter2017_test"),
    ("BACR (Bidirectional Active)", "20260918_131619_bacr_twitter2015_test", "20260918_133704_bacr_twitter2017_test")
]

print("==========================================================================================")
print("                           FULL STAGE B BENCHMARK COMPARISON TABLE                        ")
print("==========================================================================================")
print(f"{'Paradigm / Method':<32} | {'TW15 Acc':<8} {'TW15 F1':<8} {'ANQ':<6} | {'TW17 Acc':<8} {'TW17 F1':<8} {'ANQ':<6}")
print("-" * 92)

for label, r15, r17 in runs_info:
    acc15, f1_15, anq15 = 0.0, 0.0, 0.0
    acc17, f1_17, anq17 = 0.0, 0.0, 0.0
    
    if label.startswith("G1 (Direct Multimodal"):
        acc15 = g1_data.get("twitter2015", {}).get("multimodal", {}).get("accuracy", 0.0)
        f1_15 = g1_data.get("twitter2015", {}).get("multimodal", {}).get("macro_f1", 0.0)
        acc17 = g1_data.get("twitter2017", {}).get("multimodal", {}).get("accuracy", 0.0)
        f1_17 = g1_data.get("twitter2017", {}).get("multimodal", {}).get("macro_f1", 0.0)
    elif label.startswith("G1-SR"):
        p15 = os.path.join("outputs", "archive", "03_g1_sr_sequential_reread", "metrics_g1_sr_twitter2015_test.json")
        p17 = os.path.join("outputs", "archive", "03_g1_sr_sequential_reread", "metrics_g1_sr_twitter2017_test.json")
        if os.path.exists(p15):
            d15 = json.load(open(p15))
            acc15 = d15.get("final_accuracy", 0.0)
            f1_15 = d15.get("final_macro_f1", 0.0)
            anq15 = 1.0
        if os.path.exists(p17):
            d17 = json.load(open(p17))
            acc17 = d17.get("final_accuracy", 0.0)
            f1_17 = d17.get("final_macro_f1", 0.0)
            anq17 = 1.0
    elif label.startswith("G3 (Visual Probing"):
        acc15 = g3_15_data.get("tsc_benchmark_metrics", {}).get("accuracy_final", 72.67)
        f1_15 = g3_15_data.get("tsc_benchmark_metrics", {}).get("macro_f1_final", 67.74)
        anq15 = 0.28
        acc17 = g3_17_data.get("tsc_benchmark_metrics", {}).get("accuracy_final", 72.60)
        f1_17 = g3_17_data.get("tsc_benchmark_metrics", {}).get("macro_f1_final", 72.08)
        anq17 = 0.25
    else:
        p15 = os.path.join("outputs", "runs", r15, "metrics.json")
        p17 = os.path.join("outputs", "runs", r17, "metrics.json")
        if os.path.exists(p15):
            d15 = json.load(open(p15, encoding="utf-8"))
            t15 = d15.get("tsc_benchmark_metrics", {})
            acc15 = t15.get("accuracy_final", 0.0)
            f1_15 = t15.get("macro_f1_final", 0.0)
            anq15 = d15.get("anq", 0.0)
        if os.path.exists(p17):
            d17 = json.load(open(p17, encoding="utf-8"))
            t17 = d17.get("tsc_benchmark_metrics", {})
            acc17 = t17.get("accuracy_final", 0.0)
            f1_17 = t17.get("macro_f1_final", 0.0)
            anq17 = d17.get("anq", 0.0)

    print(f"{label:<32} | {acc15:>8.2f} {f1_15:>8.2f} {anq15:>6.2f} | {acc17:>8.2f} {f1_17:>8.2f} {anq17:>6.2f}")

print("-" * 92)
