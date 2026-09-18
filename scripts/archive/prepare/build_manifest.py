"""Generates canonical aspect manifests for TW15 and TW17 dev/test splits.

Produces:
- data/manifests/tw15_dev.jsonl
- data/manifests/tw15_test.jsonl
- data/manifests/tw17_dev.jsonl
- data/manifests/tw17_test.jsonl

Schema per record:
{
  "sample_id": str,
  "aspect_id": str,
  "aspect": str,
  "span": [start, end],
  "gold": "POS|NEG|NEU",
  "image": str,
  "text": str
}
"""

import os
import json
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MANIFESTS_DIR = os.path.join(PROJECT_ROOT, "data", "manifests")


SPLIT_MAPPING = {
    "tw15_dev": "twitter2015_dev.jsonl",
    "tw15_test": "twitter2015_test.jsonl",
    "tw17_dev": "twitter2017_dev.jsonl",
    "tw17_test": "twitter2017_test.jsonl",
}


def build_manifests():
    os.makedirs(MANIFESTS_DIR, exist_ok=True)

    summary = {}
    for manifest_name, src_filename in SPLIT_MAPPING.items():
        src_path = os.path.join(DATA_PROCESSED_DIR, src_filename)
        dest_path = os.path.join(MANIFESTS_DIR, f"{manifest_name}.jsonl")

        if not os.path.exists(src_path):
            print(f"[WARN] Source file not found: {src_path}")
            continue

        aspect_count = 0
        sample_count = 0
        aspect_records = []

        with open(src_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                if not line.strip():
                    continue
                data = json.loads(line)
                sid = data.get("sample_id", "unknown")
                text = data.get("text", "")
                image = data.get("image") or data.get("image_path", "")
                
                # Check annotations first
                annotations = data.get("annotations", [])
                if not annotations and "pairs" in data:
                    annotations = [{"aspect": p[0], "sentiment": p[1], "span": [0, 0]} for p in data["pairs"]]

                sample_count += 1
                for idx, ann in enumerate(annotations, 1):
                    asp_text = ann.get("aspect", "")
                    gold_sent = str(ann.get("sentiment", "NEU")).upper()
                    span = ann.get("span", [0, 0])

                    asp_record = {
                        "sample_id": sid,
                        "aspect_id": f"a_{idx:02d}",
                        "aspect": asp_text,
                        "span": span,
                        "gold": gold_sent,
                        "image": image,
                        "text": text
                    }
                    aspect_records.append(asp_record)
                    aspect_count += 1

        with open(dest_path, "w", encoding="utf-8") as f_out:
            for rec in aspect_records:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

        summary[manifest_name] = {
            "samples": sample_count,
            "aspects": aspect_count,
            "path": dest_path
        }
        print(f"[OK] {manifest_name}: {aspect_count} aspects across {sample_count} tweets -> {dest_path}")

    return summary


if __name__ == "__main__":
    build_manifests()
