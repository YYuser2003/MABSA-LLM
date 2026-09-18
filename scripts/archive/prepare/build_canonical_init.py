"""Builds canonical initial states T0 and V0 for Twitter-2015 and Twitter-2017.

Harvests text_initial (T0) and image_initial (V0) from precanonical benchmark runs,
ensuring all future experiments (G0, S0, G3, G4-TP, BACR) share the EXACT same starting point.

Outputs:
- data/cache/canonical_t0/{twitter2015_test, twitter2017_test, twitter2015_dev}.jsonl
- data/cache/canonical_t0/{tw15_test, tw17_test, tw15_dev}.jsonl (aliases)
- data/cache/canonical_v0/{twitter2015_test, twitter2017_test, twitter2015_dev}.jsonl
- data/cache/canonical_v0/{tw15_test, tw17_test, tw15_dev}.jsonl (aliases)

Format of T0:
{
  "sample_id": "...",
  "text": "...",
  "pairs": [{"aspect": "...", "sentiment": "..."}],
  "reasons": {"<aspect>": "..."},
  "aspects": [...],
  "text_initial": {...}
}
"""

import os
import json
import shutil
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CACHE_T0_DIR = os.path.join(PROJECT_ROOT, "data", "cache", "canonical_t0")
CACHE_V0_DIR = os.path.join(PROJECT_ROOT, "data", "cache", "canonical_v0")
PRECANONICAL_DIR = os.path.join(PROJECT_ROOT, "outputs", "archive", "04_g3_precanonical")

SPLIT_SOURCES = {
    "twitter2015_test": {
        "src": os.path.join(PRECANONICAL_DIR, "g3_max2_twitter2015_test.jsonl"),
        "alias": "tw15_test"
    },
    "twitter2017_test": {
        "src": os.path.join(PRECANONICAL_DIR, "g3_max2_twitter2017_test.jsonl"),
        "alias": "tw17_test"
    },
    "twitter2015_dev": {
        "src": os.path.join(PRECANONICAL_DIR, "g3_max2_twitter2015_dev.jsonl"),
        "alias": "tw15_dev"
    },
}


def build_canonical_cache():
    os.makedirs(CACHE_T0_DIR, exist_ok=True)
    os.makedirs(CACHE_V0_DIR, exist_ok=True)

    for split_key, info in SPLIT_SOURCES.items():
        src_file = info["src"]
        alias = info["alias"]

        if not os.path.exists(src_file):
            print(f"[SKIP] Source file not found: {src_file}")
            continue

        t0_dest = os.path.join(CACHE_T0_DIR, f"{split_key}.jsonl")
        t0_alias_dest = os.path.join(CACHE_T0_DIR, f"{alias}.jsonl")

        v0_dest = os.path.join(CACHE_V0_DIR, f"{split_key}.jsonl")
        v0_alias_dest = os.path.join(CACHE_V0_DIR, f"{alias}.jsonl")

        t0_count = 0
        v0_count = 0

        with open(src_file, "r", encoding="utf-8") as f_in, \
             open(t0_dest, "w", encoding="utf-8") as f_t0, \
             open(v0_dest, "w", encoding="utf-8") as f_v0:

            for line in f_in:
                if not line.strip():
                    continue
                record = json.loads(line)
                sid = record.get("sample_id")
                if not sid:
                    continue

                # Extract T0
                t_init = record.get("text_initial")
                if t_init:
                    aspects = t_init.get("aspects", [])
                    pairs = []
                    reasons = {}
                    for a in aspects:
                        asp_text = a.get("text", "")
                        sentiment = a.get("sentiment", "NEU")
                        reason = a.get("reason", "")
                        pairs.append({"aspect": asp_text, "sentiment": sentiment})
                        reasons[asp_text] = reason

                    t0_entry = {
                        "sample_id": sid,
                        "text": record.get("text", ""),
                        "pairs": pairs,
                        "reasons": reasons,
                        "aspects": aspects,
                        "text_initial": t_init
                    }
                    f_t0.write(json.dumps(t0_entry, ensure_ascii=False) + "\n")
                    t0_count += 1

                # Extract V0
                v_init = record.get("image_initial")
                if v_init:
                    v0_entry = {
                        "sample_id": sid,
                        "image": record.get("image", ""),
                        "scene": v_init.get("scene", ""),
                        "description": v_init.get("description", ""),
                        "ocr": v_init.get("ocr", []),
                        "salient_visual_cues": v_init.get("salient_visual_cues", []),
                        "possible_entities": v_init.get("possible_entities", []),
                        "image_initial": v_init
                    }
                    f_v0.write(json.dumps(v0_entry, ensure_ascii=False) + "\n")
                    v0_count += 1

        # Also write aliases (copy file)
        shutil.copyfile(t0_dest, t0_alias_dest)
        shutil.copyfile(v0_dest, v0_alias_dest)

        print(f"[OK] {split_key} ({alias}): Saved {t0_count} T0 records -> {t0_dest} & {t0_alias_dest}")
        print(f"[OK] {split_key} ({alias}): Saved {v0_count} V0 records -> {v0_dest} & {v0_alias_dest}")


if __name__ == "__main__":
    build_canonical_cache()
