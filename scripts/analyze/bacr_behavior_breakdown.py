import os
import json
from collections import defaultdict

def analyze_bacr_run(run_dir, gold_dataset_path, dataset_name):
    p_traj = os.path.join(run_dir, "trajectories.jsonl")
    p_metrics = os.path.join(run_dir, "metrics.json")
    
    # Load gold data
    gold_map = {}
    with open(gold_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            gold_map[item["sample_id"]] = item

    total_samples = 0
    total_aspects = 0
    
    total_text_queries = 0
    total_vision_queries = 0
    
    # Path sequence counters: e.g. "None", "V", "T", "T->V", "V->T", "V->V", "T->T"
    sample_seq_counts = defaultdict(int)
    
    # Aspect level tracking
    # By probe trajectory type:
    # "None", "Pure_V", "Pure_T", "T->V", "V->T"
    aspect_probe_category = defaultdict(lambda: {"total": 0, "recover": 0, "harm": 0, "stay_correct": 0, "stay_wrong": 0})
    
    # Causal attribution by updates evidence ref:
    # "vision_evidence", "text_evidence", "mixed_evidence"
    evidence_attribution = {
        "vision": {"recover": 0, "harm": 0, "net": 0},
        "text": {"recover": 0, "harm": 0, "net": 0},
        "mixed": {"recover": 0, "harm": 0, "net": 0},
        "internal_shift": {"recover": 0, "harm": 0, "net": 0} # changed without explicit evidence ref
    }
    
    detailed_updates = []
    
    with open(p_traj, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            traj = json.loads(line)
            sid = traj["sample_id"]
            if sid not in gold_map:
                continue
                
            gold_item = gold_map[sid]
            gold_aspects = gold_item["aspects"]
            gold_sentiments = gold_item["sentiments"]
            # map aspect text to gold sentiment
            gold_pairs_dict = {a.lower().strip(): s for a, s in zip(gold_aspects, gold_sentiments)}
            
            total_samples += 1
            total_aspects += len(gold_aspects)
            
            # Query counts
            n_v = traj.get("num_visual_queries", 0)
            n_t = traj.get("num_text_queries", 0)
            total_vision_queries += n_v
            total_text_queries += n_t
            
            # Sequence pattern
            rounds = traj.get("rounds", [])
            directions = [r.get("direction", "NONE").upper() for r in rounds]
            if not directions:
                seq = "No_Query (Q=0)"
            elif len(directions) == 1:
                seq = f"Single_{directions[0]}" # Single_VISUAL or Single_TEXT
            elif len(directions) == 2:
                seq = f"{directions[0]} -> {directions[1]}"
            else:
                seq = " -> ".join(directions)
                
            sample_seq_counts[seq] += 1
            
            # Initial vs Final predictions
            text_init = traj.get("text_initial", {})
            init_pairs = text_init.get("pairs", [])
            init_dict = {a.lower().strip(): s for a, s in init_pairs}
            
            final_pairs = traj.get("final_pairs", [])
            final_dict = {a.lower().strip(): s for a, s in final_pairs}
            
            # Determine trajectory classification for sample
            if n_v == 0 and n_t == 0:
                traj_cat = "Q=0 (No Query)"
            elif n_v > 0 and n_t == 0:
                traj_cat = "Pure Vision (V or V->V)"
            elif n_v == 0 and n_t > 0:
                traj_cat = "Pure Text (T or T->T)"
            else:
                if directions and len(directions) >= 2:
                    if directions[0] == "TEXT" and directions[1] == "VISUAL":
                        traj_cat = "Mixed T -> V"
                    elif directions[0] == "VISUAL" and directions[1] == "TEXT":
                        traj_cat = "Mixed V -> T"
                    else:
                        traj_cat = f"Mixed {'->'.join(directions)}"
                else:
                    traj_cat = "Mixed Other"
                    
            # Check aspect updates
            final_ctrl = traj.get("final_controller", {})
            updates_list = final_ctrl.get("updates", [])
            updates_by_aspect = {}
            for u in updates_list:
                # can match by aspect_id or text
                aid = u.get("aspect_id")
                updates_by_aspect[aid] = u
                
            init_aspects_list = text_init.get("aspects", [])
            aspect_id_to_text = {a.get("aspect_id"): a.get("text", "").lower().strip() for a in init_aspects_list}
            
            for asp, gold_s in zip(gold_aspects, gold_sentiments):
                asp_clean = asp.lower().strip()
                init_s = init_dict.get(asp_clean, "NEU")
                final_s = final_dict.get(asp_clean, "NEU")
                
                cat_dict = aspect_probe_category[traj_cat]
                cat_dict["total"] += 1
                
                is_recover = (init_s != gold_s and final_s == gold_s)
                is_harm = (init_s == gold_s and final_s != gold_s)
                
                if is_recover:
                    cat_dict["recover"] += 1
                elif is_harm:
                    cat_dict["harm"] += 1
                elif init_s == gold_s and final_s == gold_s:
                    cat_dict["stay_correct"] += 1
                else:
                    cat_dict["stay_wrong"] += 1
                    
                # If changed, analyze evidence attribution
                if init_s != final_s:
                    # find matching update
                    matched_update = None
                    for aid, u in updates_by_aspect.items():
                        if aspect_id_to_text.get(aid) == asp_clean:
                            matched_update = u
                            break
                    if not matched_update:
                        for u in updates_list:
                            if u.get("aspect_text", "").lower().strip() == asp_clean:
                                matched_update = u
                                break
                                
                    refs = matched_update.get("evidence_refs", []) if matched_update else []
                    has_vis = any("vision" in r.lower() for r in refs) or (n_v > 0 and n_t == 0)
                    has_txt = any("text" in r.lower() for r in refs) or (n_t > 0 and n_v == 0)
                    
                    target_evidence = "internal_shift"
                    if has_vis and has_txt:
                        target_evidence = "mixed"
                    elif has_vis:
                        target_evidence = "vision"
                    elif has_txt:
                        target_evidence = "text"
                        
                    if is_recover:
                        evidence_attribution[target_evidence]["recover"] += 1
                    elif is_harm:
                        evidence_attribution[target_evidence]["harm"] += 1
                        
                    detailed_updates.append({
                        "sample_id": sid,
                        "aspect": asp,
                        "gold": gold_s,
                        "from": init_s,
                        "to": final_s,
                        "is_recover": is_recover,
                        "is_harm": is_harm,
                        "traj_cat": traj_cat,
                        "evidence": target_evidence,
                        "refs": refs
                    })

    for k in evidence_attribution:
        evidence_attribution[k]["net"] = evidence_attribution[k]["recover"] - evidence_attribution[k]["harm"]

    atq_sample = total_text_queries / total_samples if total_samples else 0
    avq_sample = total_vision_queries / total_samples if total_samples else 0
    atq_aspect = total_text_queries / total_aspects if total_aspects else 0
    avq_aspect = total_vision_queries / total_aspects if total_aspects else 0
    anq_sample = (total_text_queries + total_vision_queries) / total_samples if total_samples else 0
    anq_aspect = (total_text_queries + total_vision_queries) / total_aspects if total_aspects else 0

    return {
        "dataset": dataset_name,
        "total_samples": total_samples,
        "total_aspects": total_aspects,
        "total_text_queries": total_text_queries,
        "total_vision_queries": total_vision_queries,
        "atq_sample": atq_sample,
        "avq_sample": avq_sample,
        "anq_sample": anq_sample,
        "atq_aspect": atq_aspect,
        "avq_aspect": avq_aspect,
        "anq_aspect": anq_aspect,
        "sample_seq_counts": dict(sample_seq_counts),
        "aspect_probe_category": {k: dict(v) for k, v in aspect_probe_category.items()},
        "evidence_attribution": evidence_attribution,
        "detailed_updates_count": len(detailed_updates),
        "detailed_updates": detailed_updates
    }

if __name__ == "__main__":
    res15 = analyze_bacr_run(
        "outputs/runs/20260918_131619_bacr_twitter2015_test",
        "data/processed/twitter2015_test.jsonl",
        "Twitter-2015"
    )
    res17 = analyze_bacr_run(
        "outputs/runs/20260918_133704_bacr_twitter2017_test",
        "data/processed/twitter2017_test.jsonl",
        "Twitter-2017"
    )
    
    for res in [res15, res17]:
        dname = res["dataset"]
        print(f"\n=======================================================")
        print(f"               BACR BEHAVIOR REPORT: {dname}           ")
        print(f"=======================================================")
        print(f"Total Samples: {res['total_samples']} | Total Aspects: {res['total_aspects']}")
        print(f"Total Queries: {res['total_text_queries'] + res['total_vision_queries']} (Text: {res['total_text_queries']}, Vision: {res['total_vision_queries']})")
        print(f"Sample-level:  ATQ = {res['atq_sample']:.4f}, AVQ = {res['avq_sample']:.4f}, ANQ = {res['anq_sample']:.4f}")
        print(f"Aspect-level:  ATQ = {res['atq_aspect']:.4f}, AVQ = {res['avq_aspect']:.4f}, ANQ = {res['anq_aspect']:.4f}")
        print(f"\n--- 1. Query Sequence Breakdown (Sample Level) ---")
        for seq, cnt in sorted(res["sample_seq_counts"].items(), key=lambda x: -x[1]):
            pct = cnt / res["total_samples"] * 100
            print(f"  {seq:<25}: {cnt:>4} samples ({pct:>5.2f}%)")
            
        print(f"\n--- 2. Aspect-level Performance by Probe Category ---")
        print(f"{'Probe Category':<26} | {'Aspects':<8} | {'Recover':<8} {'Harm':<8} {'Net Gain':<8} | {'Gain Rate':<9}")
        print("-" * 75)
        for cat, stats in sorted(res["aspect_probe_category"].items()):
            net = stats["recover"] - stats["harm"]
            tot = stats["total"]
            rate = f"{net/tot*100:+.2f}%" if tot else "0.0%"
            print(f"{cat:<26} | {tot:>8} | {stats['recover']:>8} {stats['harm']:>8} {net:>8} | {rate:>9}")
            
        print(f"\n--- 3. Modality Causal Attribution (Evidence-driven Recover vs Harm) ---")
        print(f"{'Modality':<18} | {'Recover (Wrong->Right)':<22} | {'Harm (Right->Wrong)':<20} | {'Net Recover':<12}")
        print("-" * 80)
        for mod, stats in res["evidence_attribution"].items():
            print(f"{mod:<18} | {stats['recover']:>22} | {stats['harm']:>20} | {stats['net']:>12}")
