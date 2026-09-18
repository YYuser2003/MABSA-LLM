import json
from collections import Counter

def analyze_dataset(ds):
    gold_file = f'data/processed/{ds}_test.jsonl'
    pred_file = f'outputs/g3_benchmark/g3_max2_{ds}_test.jsonl'
    
    gold_map = {}
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                gold_map[d['sample_id']] = d
                
    preds = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))
                
    errors = []
    corrects = []
    recoveries = []
    harms = []
    
    for p in preds:
        sid = p['sample_id']
        if sid not in gold_map:
            continue
        g = gold_map[sid]
        
        g_pairs = {x[0].lower().strip(): x[1].upper().strip() for x in g.get('pairs', [])}
        init_pairs = {x[0].lower().strip(): x[1].upper().strip() for x in p.get('text_initial', {}).get('pairs', [])}
        final_pairs = {x[0].lower().strip(): x[1].upper().strip() for x in p.get('final_pairs', [])}
        
        for asp, gold in g_pairs.items():
            init_s = init_pairs.get(asp, 'UNKNOWN')
            final_s = final_pairs.get(asp, 'UNKNOWN')
            
            item = {
                'sid': sid,
                'text': g.get('text', ''),
                'image': g.get('image', ''),
                'aspect': asp,
                'gold': gold,
                'init': init_s,
                'final': final_s,
                'queries': [r.get('query_type') for r in p.get('rounds', [])],
                'reasoning': p.get('final_controller', {}).get('reasoning', '')
            }
            if final_s == gold:
                corrects.append(item)
                if init_s != gold:
                    recoveries.append(item)
            else:
                errors.append(item)
                if init_s == gold:
                    harms.append(item)
                    
    conf = Counter((e['gold'], e['final']) for e in errors)
    return {
        'total_gold': len(corrects) + len(errors),
        'correct': len(corrects),
        'errors_count': len(errors),
        'recoveries_count': len(recoveries),
        'harms_count': len(harms),
        'confusion': conf,
        'errors': errors,
        'recoveries': recoveries
    }

res15 = analyze_dataset('twitter2015')
res17 = analyze_dataset('twitter2017')

print(f"=== TWITTER2015 ===")
print(f"Total Aspects: {res15['total_gold']}, Correct: {res15['correct']} ({res15['correct']/res15['total_gold']*100:.2f}%), Errors: {res15['errors_count']}")
print(f"Aspect Recoveries (W->C): {res15['recoveries_count']}, Aspect Harms (C->W): {res15['harms_count']}")
print("Confusion Matrix for Errors:")
for k, v in res15['confusion'].most_common():
    print(f"  Gold {k[0]:3s} -> Pred {k[1]:3s}: {v:3d} ({v/res15['errors_count']*100:5.1f}%)")

print(f"\n=== TWITTER2017 ===")
print(f"Total Aspects: {res17['total_gold']}, Correct: {res17['correct']} ({res17['correct']/res17['total_gold']*100:.2f}%), Errors: {res17['errors_count']}")
print(f"Aspect Recoveries (W->C): {res17['recoveries_count']}, Aspect Harms (C->W): {res17['harms_count']}")
print("Confusion Matrix for Errors:")
for k, v in res17['confusion'].most_common():
    print(f"  Gold {k[0]:3s} -> Pred {k[1]:3s}: {v:3d} ({v/res17['errors_count']*100:5.1f}%)")

print("\n" + "="*80)
print("SAMPLE CITATIONS ACROSS CATEGORIES")
print("="*80)

def find_examples(res, ds_name):
    print(f"\n--- {ds_name} EXAMPLES ---")
    types = [('NEU', 'POS'), ('NEU', 'NEG'), ('POS', 'NEU'), ('NEG', 'NEU'), ('POS', 'NEG'), ('NEG', 'POS')]
    for g_t, f_t in types:
        matched = [e for e in res['errors'] if e['gold'] == g_t and e['final'] == f_t]
        if matched:
            ex = matched[0]
            print(f"[{g_t} -> {f_t}] ({len(matched)} total) ID: {ex['sid']}")
            print(f"  Text: {ex['text']}")
            print(f"  Aspect: {ex['aspect']}")
            print(f"  Init: {ex['init']} -> Final: {ex['final']} (Gold: {ex['gold']})")
            print(f"  Queries: {ex['queries']}")

find_examples(res15, "TWITTER-2015")
find_examples(res17, "TWITTER-2017")
