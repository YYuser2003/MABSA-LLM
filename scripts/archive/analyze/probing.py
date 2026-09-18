"""
Analysis and Reporting for Qwen3-VL-8B Visual Entity & Celebrity Recognition.
Aggregates predictions across Twitter-2015 and Twitter-2017 (Dev & Test).
Computes:
1. Aspect Visibility Rate (Aspect-level Grounding Rate).
2. Entity Type Distribution (Person vs Organization vs Location vs Other).
3. Celebrity Recognition Rate on Ground-truth Person Aspects.
4. Per-dataset and Cross-dataset Comparisons.
"""

import os
import json
import re
from collections import defaultdict, Counter
from typing import Dict, Any, List

def normalize_name(name: str) -> str:
    if not name:
        return ""
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', name).lower().strip()
    return clean

def is_name_match(aspect: str, recognized: str) -> bool:
    if not recognized or recognized.lower() in ["unknown", "none", "n/a"]:
        return False
    a_norm = normalize_name(aspect)
    r_norm = normalize_name(recognized)
    if not a_norm or not r_norm:
        return False
    # Exact match
    if a_norm == r_norm:
        return True
    # Substring containment: e.g. "trump" in "donald trump" or "kanye" in "kanye west"
    if a_norm in r_norm or r_norm in a_norm:
        return True
    # Token overlap
    a_tokens = set(a_norm.split())
    r_tokens = set(r_norm.split())
    if a_tokens & r_tokens:
        return True
    return False

def analyze_probing_results(output_dir: str):
    splits = ["dev", "test"]
    datasets = ["twitter2015", "twitter2017"]
    
    all_aspects = []
    
    for ds in datasets:
        for sp in splits:
            fpath = os.path.join(output_dir, f"entity_recognition_{ds}_{sp}_preds.jsonl")
            if not os.path.exists(fpath):
                print(f"[Notice] File not found: {fpath}")
                continue
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line.strip())
                    sid = item.get("sample_id", "")
                    gold_pairs = item.get("gold_pairs", [])
                    aspects_gold = [p[0] for p in gold_pairs]
                    preds = item.get("visual_entity_predictions", [])
                    
                    # Map prediction by aspect name
                    pred_map = {}
                    for p in preds:
                        if isinstance(p, dict) and "aspect" in p:
                            pred_map[normalize_name(p["aspect"])] = p
                            
                    for asp in aspects_gold:
                        a_norm = normalize_name(asp)
                        pred = pred_map.get(a_norm, {})
                        
                        is_vis = pred.get("is_visible", False)
                        cat = pred.get("category", "other").lower() if pred else "other"
                        rec_name = pred.get("recognized_name", "Unknown") if pred else "Unknown"
                        cues = pred.get("visual_evidence", "") if pred else ""
                        
                        match = is_name_match(asp, rec_name)
                        
                        all_aspects.append({
                            "sample_id": sid,
                            "dataset": ds,
                            "split": sp,
                            "aspect": asp,
                            "is_visible": is_vis,
                            "category": cat,
                            "recognized_name": rec_name,
                            "visual_evidence": cues,
                            "is_celebrity_recognized": match
                        })
                        
    total_n = len(all_aspects)
    if total_n == 0:
        print("No evaluated aspect records found.")
        return
        
    print("\n" + "=" * 92)
    print("      QWEN3-VL-8B VISUAL ENTITY & CELEBRITY RECOGNITION REPORT (ALL TW15 & TW17)")
    print("=" * 92)
    print(f"Total Evaluated Aspects: {total_n}")
    
    # 1. Dataset Breakdown
    print("\n1. Overall Visual Grounding & Visibility by Dataset:")
    print(f"{'Subset':25s} | {'N Aspects':10s} | {'Physically Visible':20s} | {'Visibility %':12s}")
    print("-" * 75)
    for ds in datasets:
        for sp in splits:
            sub = [x for x in all_aspects if x["dataset"] == ds and x["split"] == sp]
            n_sub = len(sub)
            if n_sub == 0:
                continue
            vis_cnt = sum(1 for x in sub if x["is_visible"])
            pct = vis_cnt / n_sub * 100
            label = f"{ds.upper()} ({sp})"
            print(f"{label:25s} | {n_sub:10d} | {vis_cnt:10d} / {n_sub:<7d} | {pct:10.2f}%")
            
    vis_all = sum(1 for x in all_aspects if x["is_visible"])
    print("-" * 75)
    print(f"{'TOTAL (All 15 & 17)':25s} | {total_n:10d} | {vis_all:10d} / {total_n:<7d} | {vis_all/total_n*100:10.2f}%")
    
    # 2. Entity Category Distribution
    print("\n2. Predicted Entity Category Distribution:")
    cat_counts = Counter(x["category"] for x in all_aspects)
    for cat, cnt in cat_counts.most_common():
        print(f"   - {cat:15s}: {cnt:5d} ({cnt/total_n*100:5.2f}%)")
        
    # 3. Celebrity Recognition on Person Aspects
    person_aspects = [x for x in all_aspects if x["category"] == "person"]
    n_person = len(person_aspects)
    print(f"\n3. Celebrity Recognition Performance on Person Aspects (N = {n_person}):")
    if n_person > 0:
        vis_person = [x for x in person_aspects if x["is_visible"]]
        n_vis_person = len(vis_person)
        rec_correct = sum(1 for x in person_aspects if x["is_celebrity_recognized"])
        rec_vis_correct = sum(1 for x in vis_person if x["is_celebrity_recognized"])
        
        print(f"   - Person Aspects Visible in Image:     {n_vis_person} / {n_person} ({n_vis_person/n_person*100:.2f}%)")
        print(f"   - Celebrity Recognition (Overall):     {rec_correct} / {n_person} ({rec_correct/n_person*100:.2f}%)")
        if n_vis_person > 0:
            print(f"   - Celebrity Recognition (When Visible): {rec_vis_correct} / {n_vis_person} ({rec_vis_correct/n_vis_person*100:.2f}%)")
            
        print("\n4. Top Recognized Public Figures & Celebrities:")
        rec_figures = [x["recognized_name"] for x in person_aspects if x["is_celebrity_recognized"]]
        for name, cnt in Counter(rec_figures).most_common(15):
            print(f"   - {name:25s}: recognized {cnt:3d} times")
            
        print("\n5. Person Aspect Failure Case Analysis:")
        # Visible person but unrecognized (Blindspot)
        blindspots = [x for x in vis_person if not x["is_celebrity_recognized"] and x["recognized_name"].lower() in ["unknown", "none", "n/a", ""]]
        # Hallucinated or mismatched person
        mismatches = [x for x in vis_person if not x["is_celebrity_recognized"] and x["recognized_name"].lower() not in ["unknown", "none", "n/a", ""]]
        
        print(f"   - Recognition Blindspots (Visible person, but name Unknown): {len(blindspots)} ({len(blindspots)/n_vis_person*100:.2f}%)")
        print(f"   - Mismatches / Misidentifications (Identified as different name): {len(mismatches)} ({len(mismatches)/n_vis_person*100:.2f}%)")
        
        if blindspots:
            print("\n   [Typical Recognition Blindspot Examples (Seen but Unnamed)]:")
            for item in blindspots[:3]:
                print(f"     * Aspect: '{item['aspect']}' | Evidence: '{item['visual_evidence']}' | Sample: {item['sample_id']}")
                
        if mismatches:
            print("\n   [Typical Misidentification Examples]:")
            for item in mismatches[:3]:
                print(f"     * Target: '{item['aspect']}' -> Predicted: '{item['recognized_name']}' | Sample: {item['sample_id']}")

    print("=" * 92 + "\n")

if __name__ == '__main__':
    base_dir = "/opt/data/private/MABSA-LLM"
    out_dir = os.path.join(base_dir, "outputs/visual_probing")
    analyze_probing_results(out_dir)
