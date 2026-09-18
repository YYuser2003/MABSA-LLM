import json
from collections import defaultdict

def analyze_g3_deep(ds_name, pred_file, gold_file):
    gold_map = {}
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                gold_map[d['sample_id']] = d

    samples = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Buckets by Q
    q_buckets = {0: [], 1: [], 2: []}
    
    # Global counters for probabilities
    # Sample-level
    s_init_wrong_total = 0
    s_init_wrong_queried = 0
    s_init_corr_total = 0
    s_init_corr_queried = 0
    s_queried_wrong_recovered = 0
    s_queried_corr_harmed = 0
    
    # Aspect-level
    a_init_wrong_total = 0
    a_init_wrong_queried = 0
    a_init_corr_total = 0
    a_init_corr_queried = 0
    a_queried_wrong_recovered = 0
    a_queried_corr_harmed = 0

    for s in samples:
        sid = s['sample_id']
        if sid not in gold_map:
            continue
        g = gold_map[sid]
        
        g_pairs = {x[0].lower().strip(): x[1].upper().strip() for x in g.get('pairs', [])}
        init_pairs = {x[0].lower().strip(): x[1].upper().strip() for x in s.get('text_initial', {}).get('pairs', [])}
        final_pairs = {x[0].lower().strip(): x[1].upper().strip() for x in s.get('final_pairs', [])}
        
        num_q = s.get('num_visual_queries', 0)
        num_q_clamped = min(num_q, 2)
        
        # Determine sample-level correctness
        s_init_corr = all(init_pairs.get(a) == gold for a, gold in g_pairs.items())
        s_final_corr = all(final_pairs.get(a) == gold for a, gold in g_pairs.items())
        is_queried = (num_q > 0)
        
        if s_init_corr:
            s_init_corr_total += 1
            if is_queried:
                s_init_corr_queried += 1
                if not s_final_corr:
                    s_queried_corr_harmed += 1
        else:
            s_init_wrong_total += 1
            if is_queried:
                s_init_wrong_queried += 1
                if s_final_corr:
                    s_queried_wrong_recovered += 1

        # Aspect level
        asp_data = []
        for asp, gold in g_pairs.items():
            i_s = init_pairs.get(asp, 'UNKNOWN')
            f_s = final_pairs.get(asp, 'UNKNOWN')
            i_ok = (i_s == gold)
            f_ok = (f_s == gold)
            
            if i_ok:
                a_init_corr_total += 1
                if is_queried:
                    a_init_corr_queried += 1
                    if not f_ok:
                        a_queried_corr_harmed += 1
            else:
                a_init_wrong_total += 1
                if is_queried:
                    a_init_wrong_queried += 1
                    if f_ok:
                        a_queried_wrong_recovered += 1
                        
            asp_data.append((asp, gold, i_s, f_s, i_ok, f_ok))
            
        q_buckets[num_q_clamped].append({
            'sid': sid,
            'aspects': asp_data,
            's_init_corr': s_init_corr,
            's_final_corr': s_final_corr
        })

    def eval_aspect_list(asp_tuples):
        # asp_tuples: list of (gold, pred)
        if not asp_tuples:
            return 0.0, 0.0
        acc = sum(1 for g, p in asp_tuples if g == p) / len(asp_tuples) * 100.0
        classes = ['POS', 'NEG', 'NEU']
        f1s = []
        for cls in classes:
            tp = sum(1 for g, p in asp_tuples if g == cls and p == cls)
            fp = sum(1 for g, p in asp_tuples if g != cls and p == cls)
            fn = sum(1 for g, p in asp_tuples if g == cls and p != cls)
            p = tp / (tp + fp) if tp + fp > 0 else 0.0
            r = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
            f1s.append(f1)
        mf1 = sum(f1s) / len(f1s) * 100.0
        return round(acc, 2), round(mf1, 2)

    print(f"\n================================================================================")
    print(f"                   DEEP G3 DIAGNOSTICS: {ds_name.upper()}")
    print(f"================================================================================")
    
    print("\n--- 1. STRATIFIED SUBSET ANALYSIS (Q = 0, Q = 1, Q = 2) ---")
    print(f"{'Subset':<8} | {'Samples':<8} | {'Aspects':<8} | {'Init Acc':<9} | {'Init MF1':<9} | {'Final Acc':<10} | {'Final MF1':<10} | {'Delta Acc':<10} | {'Delta MF1':<10}")
    print("-" * 95)
    
    for q_val in [0, 1, 2]:
        b_samples = q_buckets[q_val]
        n_s = len(b_samples)
        all_init_asp = []
        all_final_asp = []
        for s in b_samples:
            for asp, gold, i_s, f_s, i_ok, f_ok in s['aspects']:
                all_init_asp.append((gold, i_s))
                all_final_asp.append((gold, f_s))
                
        n_a = len(all_init_asp)
        i_acc, i_mf1 = eval_aspect_list(all_init_asp)
        f_acc, f_mf1 = eval_aspect_list(all_final_asp)
        d_acc = round(f_acc - i_acc, 2)
        d_mf1 = round(f_mf1 - i_mf1, 2)
        print(f"Q = {q_val:<4} | {n_s:<8} | {n_a:<8} | {i_acc:<9.2f} | {i_mf1:<9.2f} | {f_acc:<10.2f} | {f_mf1:<10.2f} | {d_acc:<+10.2f} | {d_mf1:<+10.2f}")

    print("\n--- 2. CRITICAL CONDITIONAL PROBABILITIES ---")
    p_q_given_w_sample = (s_init_wrong_queried / s_init_wrong_total * 100.0) if s_init_wrong_total > 0 else 0.0
    p_q_given_c_sample = (s_init_corr_queried / s_init_corr_total * 100.0) if s_init_corr_total > 0 else 0.0
    p_rec_given_qw_sample = (s_queried_wrong_recovered / s_init_wrong_queried * 100.0) if s_init_wrong_queried > 0 else 0.0
    p_harm_given_qc_sample = (s_queried_corr_harmed / s_init_corr_queried * 100.0) if s_init_corr_queried > 0 else 0.0
    
    p_q_given_w_aspect = (a_init_wrong_queried / a_init_wrong_total * 100.0) if a_init_wrong_total > 0 else 0.0
    p_q_given_c_aspect = (a_init_corr_queried / a_init_corr_total * 100.0) if a_init_corr_total > 0 else 0.0
    p_rec_given_qw_aspect = (a_queried_wrong_recovered / a_init_wrong_queried * 100.0) if a_init_wrong_queried > 0 else 0.0
    p_harm_given_qc_aspect = (a_queried_corr_harmed / a_init_corr_queried * 100.0) if a_init_corr_queried > 0 else 0.0

    print("Sample-Level Behavior:")
    print(f"  P(query | initial wrong)               : {p_q_given_w_sample:.2f}%  ({s_init_wrong_queried} / {s_init_wrong_total})")
    print(f"  P(query | initial correct)             : {p_q_given_c_sample:.2f}%  ({s_init_corr_queried} / {s_init_corr_total})")
    print(f"  P(recover | queried and initial wrong) : {p_rec_given_qw_sample:.2f}%  ({s_queried_wrong_recovered} / {s_init_wrong_queried})")
    print(f"  P(harm | queried and initial correct)  : {p_harm_given_qc_sample:.2f}%  ({s_queried_corr_harmed} / {s_init_corr_queried})")

    print("\nAspect-Level Behavior:")
    print(f"  P(query | initial wrong)               : {p_q_given_w_aspect:.2f}%  ({a_init_wrong_queried} / {a_init_wrong_total})")
    print(f"  P(query | initial correct)             : {p_q_given_c_aspect:.2f}%  ({a_init_corr_queried} / {a_init_corr_total})")
    print(f"  P(recover | queried and initial wrong) : {p_rec_given_qw_aspect:.2f}%  ({a_queried_wrong_recovered} / {a_init_wrong_queried})")
    print(f"  P(harm | queried and initial correct)  : {p_harm_given_qc_aspect:.2f}%  ({a_queried_corr_harmed} / {a_init_corr_queried})")

analyze_g3_deep('twitter2015', 'outputs/g3_benchmark/g3_max2_twitter2015_test.jsonl', 'data/processed/twitter2015_test.jsonl')
analyze_g3_deep('twitter2017', 'outputs/g3_benchmark/g3_max2_twitter2017_test.jsonl', 'data/processed/twitter2017_test.jsonl')
