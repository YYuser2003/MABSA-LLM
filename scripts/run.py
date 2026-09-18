"""Master Experiment Execution Runner for BACR / MABSA-LLM.

Configuration-driven entry point:
Usage:
  python scripts/run.py --config configs/experiments/bacr.yaml --dataset twitter2017 --split dev
  python scripts/run.py --config configs/experiments/g3_visual_only.yaml --dataset twitter2015 --split test

Core Philosophy:
  Code by responsibility, experiments by config, results by run.
"""

import os
import sys
import json
import time
import yaml
import random
import argparse
import logging
import threading
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Set

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bacr.client import GeminiClient
from bacr.pipeline import BACRPipeline, G3Pipeline
from bacr.pipeline_v3 import BACRPipelineV3
from bacr.evaluator import (
    BACREvaluator,
    print_evaluation_report,
    evaluate_v3_trajectories,
    print_v3_evaluation_report
)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), ""), content)
    return yaml.safe_load(content)


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("BACR_Runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File Handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def main():
    parser = argparse.ArgumentParser(description="Master Experiment Runner for BACR / MABSA-LLM.")
    parser.add_argument("--config", "-c", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--dataset", choices=["twitter2015", "twitter2017"], default=None, help="Override target dataset.")
    parser.add_argument("--split", choices=["test", "dev", "train"], default=None, help="Override dataset split.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--concurrency", type=int, default=None, help="Override concurrency workers.")
    parser.add_argument("--tag", type=str, default=None, help="Custom run tag.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resume", type=str, default=None, help="Resume run from existing run directory or run ID.")
    parser.add_argument("--allow-partial-eval", action="store_true", help="Allow evaluation even if some samples failed.")
    args = parser.parse_args()

    # 1. Load experiment config
    exp_cfg_path = os.path.abspath(args.config) if os.path.exists(args.config) else os.path.join(PROJECT_ROOT, args.config)
    exp_config = load_yaml(exp_cfg_path)
    exp_name = exp_config.get("experiment", {}).get("name", "experiment")

    # 2. Resolve dataset config
    dataset_name = args.dataset or "twitter2015"
    ds_cfg_file = "twitter15.yaml" if "15" in dataset_name else "twitter17.yaml"
    ds_cfg_path = os.path.join(PROJECT_ROOT, "configs", "datasets", ds_cfg_file)
    ds_config = load_yaml(ds_cfg_path)

    split = args.split or "test"
    concurrency = args.concurrency or exp_config.get("experiment", {}).get("concurrency", 4)

    # 3. Resolve Model config
    model_cfg_path = os.path.join(PROJECT_ROOT, exp_config.get("experiment", {}).get("model_config", "configs/models/gemini.yaml"))
    model_config = load_yaml(model_cfg_path)

    # 4. Resolve run_id and run_dir (support resume)
    if args.resume:
        if os.path.isdir(args.resume):
            run_dir = os.path.abspath(args.resume)
        elif os.path.isdir(os.path.join(PROJECT_ROOT, "outputs", "runs", args.resume)):
            run_dir = os.path.join(PROJECT_ROOT, "outputs", "runs", args.resume)
        else:
            raise FileNotFoundError(f"Cannot find run dir for resume: {args.resume}")
        run_id = os.path.basename(run_dir)
    else:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_str = f"_{args.tag}" if args.tag else ""
        run_id = f"{date_str}_{exp_name}_{dataset_name}_{split}{tag_str}"
        run_dir = os.path.join(PROJECT_ROOT, "outputs", "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "run.log")
    logger = setup_logger(log_file)

    logger.info("=" * 70)
    logger.info("   BACR EXPERIMENT RUNNER (CONFIGURATION-DRIVEN)")
    logger.info("=" * 70)
    logger.info(f"Run ID          : {run_id}")
    logger.info(f"Experiment      : {exp_name}")
    logger.info(f"Dataset / Split : {dataset_name} ({split})")
    logger.info(f"Concurrency     : {concurrency}")
    logger.info(f"Output Run Dir  : {run_dir}")
    if args.resume:
        logger.info(f"Resume Mode     : Active (resuming from {run_dir})")

    # 5. Save snapshot of full resolved configuration
    full_resolved_config = {
        "run_id": run_id,
        "date": datetime.now().isoformat(),
        "experiment": exp_config,
        "dataset": ds_config,
        "model": model_config,
        "cli_args": vars(args)
    }
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f_cfg:
        yaml.dump(full_resolved_config, f_cfg, default_flow_style=False, allow_unicode=True)

    # 6. Locate dataset file & manifest
    data_file = os.path.join(PROJECT_ROOT, ds_config.get("dataset", {}).get("splits", {}).get(split))
    manifest_prefix = ds_config.get("dataset", {}).get("manifest_prefix", "tw15")
    manifest_file = os.path.join(PROJECT_ROOT, "data", "manifests", f"{manifest_prefix}_{split}.jsonl")

    # Load canonical caches independently
    t0_cache = {}
    v0_cache = {}
    use_canonical_t0 = exp_config.get("experiment", {}).get("canonical_t0", False)
    use_canonical_v0 = exp_config.get("experiment", {}).get("canonical_v0", False)

    if use_canonical_t0:
        t0_candidates = [
            os.path.join(PROJECT_ROOT, "data", "cache", "canonical_t0", f"{manifest_prefix}_{split}.jsonl"),
            os.path.join(PROJECT_ROOT, "data", "cache", "canonical_t0", f"{dataset_name}_{split}.jsonl")
        ]
        t0_path = next((p for p in t0_candidates if os.path.exists(p)), t0_candidates[0])
        if os.path.exists(t0_path):
            with open(t0_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        t0_cache[d["sample_id"]] = d
            logger.info(f"Loaded {len(t0_cache)} canonical T0 initial records from {os.path.basename(t0_path)}.")

    if use_canonical_v0:
        v0_candidates = [
            os.path.join(PROJECT_ROOT, "data", "cache", "canonical_v0", f"{manifest_prefix}_{split}.jsonl"),
            os.path.join(PROJECT_ROOT, "data", "cache", "canonical_v0", f"{dataset_name}_{split}.jsonl")
        ]
        v0_path = next((p for p in v0_candidates if os.path.exists(p)), v0_candidates[0])
        if os.path.exists(v0_path):
            with open(v0_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        v0_cache[d["sample_id"]] = d
            logger.info(f"Loaded {len(v0_cache)} canonical V0 initial records from {os.path.basename(v0_path)}.")

    # Load input samples
    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if args.limit and len(samples) >= args.limit:
                    break

    logger.info(f"Total target samples to process: {len(samples)}")

    pred_file = os.path.join(run_dir, "predictions.jsonl")
    traj_file = os.path.join(run_dir, "trajectories.jsonl")

    # Check for already processed samples (useful for resume)
    existing_sample_ids = set()
    if os.path.exists(traj_file):
        with open(traj_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        if "sample_id" in d:
                            existing_sample_ids.add(d["sample_id"])
                    except Exception:
                        pass
        if existing_sample_ids:
            logger.info(f"Found {len(existing_sample_ids)} already processed samples in {os.path.basename(traj_file)}.")

    exp_type = exp_config.get("experiment", {}).get("type", "")
    if exp_type == "text_only" and use_canonical_t0:
        logger.info("Executing G0 / T0 Text Initial Reasoner evaluation from Canonical T0 cache (0 API calls)...")
        for sample in samples:
            sid = sample.get("sample_id", "unknown")
            t0 = t0_cache.get(sid, {})
            # extract pairs
            pairs = []
            if "pairs" in t0 and isinstance(t0["pairs"], list):
                if t0["pairs"] and isinstance(t0["pairs"][0], dict):
                    pairs = [[p["aspect"], p["sentiment"]] for p in t0["pairs"]]
                else:
                    pairs = t0["pairs"]
            elif "text_initial" in t0 and isinstance(t0["text_initial"], dict):
                pairs = t0["text_initial"].get("pairs", [])
            elif "aspects" in t0 and isinstance(t0["aspects"], list):
                pairs = [[a.get("text", a.get("aspect", "")), a.get("sentiment", "NEU")] for a in t0["aspects"]]

            traj_entry = {
                "sample_id": sid,
                "text": sample.get("text", ""),
                "text_initial": t0.get("text_initial", t0),
                "final_pairs": pairs,
                "num_visual_queries": 0,
                "stop_type": "canonical_t0_direct"
            }
            pred_entry = {
                "sample_id": sid,
                "predictions": pairs,
                "stop_type": "canonical_t0_direct",
                "num_visual_queries": 0
            }
            with open(traj_file, "a", encoding="utf-8") as f_traj:
                f_traj.write(json.dumps(traj_entry, ensure_ascii=False) + "\n")
            with open(pred_file, "a", encoding="utf-8") as f_pred:
                f_pred.write(json.dumps(pred_entry, ensure_ascii=False) + "\n")
    else:
        # Filter remaining samples if resuming
        samples_to_run = [s for s in samples if s.get("sample_id") not in existing_sample_ids]
        logger.info(f"Target samples remaining to process: {len(samples_to_run)} / {len(samples)}")

        # Initialize client & pipeline
        def create_client(cfg: Dict[str, Any]):
            m_info = cfg.get("model", cfg)
            provider = cfg.get("provider", m_info.get("provider", "gemini")).lower()
            if provider in ["qwen", "openai", "vllm"]:
                from bacr.client import OpenAICompatibleClient
                return OpenAICompatibleClient(
                    model=m_info.get("name", "Qwen/Qwen2.5-VL-7B-Instruct"),
                    base_url=m_info.get("base_url", "http://localhost:8000/v1"),
                    api_key=m_info.get("api_key", "EMPTY"),
                    temperature=float(m_info.get("temperature", 0.1))
                )
            else:
                return GeminiClient(
                    model=m_info.get("name", "gemini_3.8"),
                    base_url=m_info.get("base_url"),
                    thinking_level=m_info.get("thinking_level", "high"),
                    temperature=float(m_info.get("temperature", 0.1)),
                    allow_fallback=m_info.get("allow_fallback", False)
                )

        client = create_client(model_config)

        # Check for role-specific clients (e.g. Qwen for controller, Gemini for text)
        roles_cfg = exp_config.get("experiment", {}).get("roles", {})
        text_client = create_client(roles_cfg["text"]) if "text" in roles_cfg else client
        controller_client = create_client(roles_cfg["controller"]) if "controller" in roles_cfg else client
        vision_client = create_client(roles_cfg["vision"]) if "vision" in roles_cfg else client

        exp_ver = exp_config.get("experiment", {}).get("version", "") or exp_config.get("experiment", {}).get("type", "")
        is_v3 = (exp_ver == "bacr_v3" or "v3" in exp_name.lower())

        failures_file = os.path.join(run_dir, "failures.jsonl")

        if is_v3:
            budget_cfg = exp_config.get("experiment", {}).get("budget", {})
            max_deep_actions = budget_cfg.get("max_deep_actions", exp_config.get("experiment", {}).get("max_visual_probes", 2))
            task_mode = exp_config.get("experiment", {}).get("task", {}).get("mode", "target_guided")
            pipeline = BACRPipelineV3(
                client=client,
                run_id=run_id,
                max_visual_probes=max_deep_actions,
                budget_config=budget_cfg,
                task_mode=task_mode,
                text_client=text_client,
                controller_client=controller_client,
                vision_client=vision_client
            )
            logger.info(f"Initialized BACRPipelineV3 (Task Mode={task_mode}, Budget={budget_cfg})")
        else:
            controller_sees_raw = exp_config.get("experiment", {}).get("controller", {}).get("controller_sees_raw_modalities", False)
            pipeline = BACRPipeline(client=client, controller_sees_raw_modalities=controller_sees_raw)
            logger.info(f"Initialized BACRPipeline (Controller Sees Raw={controller_sees_raw})")

        file_lock = threading.Lock()
        processed_count = len(existing_sample_ids)
        total_count = len(samples)
        start_time = time.time()
        max_queries = exp_config.get("experiment", {}).get("controller", {}).get("max_queries", 2)
        allowed_directions = exp_config.get("experiment", {}).get("controller", {}).get("directions", None)

        def process_single(sample: Dict[str, Any]):
            nonlocal processed_count
            sid = sample.get("sample_id", "unknown")
            # Check canonical injection
            sample_copy = dict(sample)
            if sid in t0_cache:
                sample_copy["text_initial_cached"] = t0_cache[sid]
            if sid in v0_cache:
                sample_copy["image_initial_cached"] = v0_cache[sid]

            try:
                if is_v3:
                    record = pipeline.run_sample(
                        sample=sample_copy,
                        image_base_dir=PROJECT_ROOT,
                        max_visual_probes=max_deep_actions
                    )
                else:
                    record = pipeline.run_sample(
                        sample=sample_copy,
                        image_base_dir=PROJECT_ROOT,
                        max_queries=max_queries,
                        allowed_directions=allowed_directions
                    )
                with file_lock:
                    with open(traj_file, "a", encoding="utf-8") as f_traj:
                        f_traj.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # Format predictions
                    num_v = record.get("num_visual_probes", record.get("num_visual_queries", 0))
                    num_t = record.get("num_text_queries", 0)
                    pred_entry = {
                        "sample_id": sid,
                        "predictions": record.get("final_pairs", []),
                        "stop_type": record.get("initial_contrast", {}).get("action", record.get("stop_type", "natural_stop")),
                        "num_visual_queries": num_v,
                        "num_text_queries": num_t,
                        "num_queries_total": num_v + num_t
                    }
                    with open(pred_file, "a", encoding="utf-8") as f_pred:
                        f_pred.write(json.dumps(pred_entry, ensure_ascii=False) + "\n")

                    processed_count += 1
                    elapsed = time.time() - start_time
                    if is_v3:
                        c_type = record.get("contrast_type", "UNKNOWN")
                        logger.info(f"[{processed_count}/{total_count}] Success: {sid} (Contrast={c_type}, Probes={num_v}, Elapsed: {elapsed:.1f}s)")
                    else:
                        logger.info(f"[{processed_count}/{total_count}] Success: {sid} (Queries: V={num_v}/T={num_t}, Elapsed: {elapsed:.1f}s)")
                return True
            except Exception as e:
                logger.error(f"[ERROR] Failed {sid}: {e}")
                with file_lock:
                    with open(failures_file, "a", encoding="utf-8") as f_fail:
                        f_fail.write(json.dumps({"sample_id": sid, "error": str(e)}, ensure_ascii=False) + "\n")
                return False

        if samples_to_run:
            logger.info(f"Starting execution pool with {concurrency} workers for {len(samples_to_run)} samples...")
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(process_single, s) for s in samples_to_run]
                for fut in as_completed(futures):
                    fut.result()
        else:
            logger.info("All target samples already processed!")

        # Failure check
        failures = []
        if os.path.exists(failures_file):
            with open(failures_file, "r", encoding="utf-8") as f_fail:
                for line in f_fail:
                    if line.strip():
                        failures.append(json.loads(line))
        if failures and not getattr(args, "allow_partial_eval", False):
            logger.error(f"INCOMPLETE RUN: {len(failures)} samples failed. Evaluation blocked. Pass --allow-partial-eval to override.")
            sys.exit(1)

    if os.path.exists(traj_file) and os.path.getsize(traj_file) > 0:
        # Deduplicate trajectories by sample_id keeping latest entry
        deduped_trajs = {}
        with open(traj_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        sid = d.get("sample_id")
                        if sid:
                            deduped_trajs[sid] = d
                    except Exception:
                        pass
        with open(traj_file, "w", encoding="utf-8") as f:
            for d in deduped_trajs.values():
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        # Also dedup predictions
        if os.path.exists(pred_file):
            deduped_preds = {}
            with open(pred_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            sid = d.get("sample_id")
                            if sid:
                                deduped_preds[sid] = d
                        except Exception:
                            pass
            with open(pred_file, "w", encoding="utf-8") as f:
                for d in deduped_preds.values():
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

        logger.info(f"Inference complete! Evaluating {len(deduped_trajs)} trajectories...")
        evaluator = BACREvaluator(gold_file=data_file)
        metrics = evaluator.evaluate_predictions(pred_file=traj_file)
        print_evaluation_report(metrics)

        # If v3, also compute and display dual-hypothesis contrast diagnostics
        if is_v3:
            v3_diag = evaluate_v3_trajectories(traj_file=traj_file, gold_file=data_file)
            print_v3_evaluation_report(v3_diag)
            v3_diag_path = os.path.join(run_dir, "v3_diagnostics.json")
            with open(v3_diag_path, "w", encoding="utf-8") as f_v3:
                json.dump(v3_diag, f_v3, indent=2, ensure_ascii=False)

        # Save metrics
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f_m:
            json.dump(metrics, f_m, indent=2, ensure_ascii=False)
        logger.info(f"Experiment {run_id} successfully finished! Artifacts saved in: {run_dir}")
    else:
        logger.warning(f"No trajectory outputs generated in {traj_file}. Evaluation skipped.")


if __name__ == "__main__":
    main()
