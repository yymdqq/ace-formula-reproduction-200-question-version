"""Run and summarize the official ACE offline Formula experiment on fixed subsets."""

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "eval" / "finance" / "data"
UPSTREAM_COMMIT = "82709de050e1db6e6ef2f07bcb0393560b94992a"
SPLITS = {
    "train": "formula_train_subset_500.jsonl",
    "val": "formula_val_subset_300.jsonl",
    "test": "formula_test.jsonl",
}
CONDITIONS = {"baseline": 5, "short_reflection": 1}
PAPER_PROFILES = {
    "paper": ((None, None, None), 5),
    "paper-subset": ((100, 50, None), 5),
    "paper-short": ((24, 8, 30), 2),
}
METRICS = (
    "empty_playbook_accuracy_pct", "final_accuracy_pct", "accuracy_gain_pp",
    "wall_seconds", "api_calls", "reflection_retry_calls", "input_tokens", "output_tokens",
)
BASE_LLM_METRICS = ("base_llm_accuracy_pct", "ace_minus_base_llm_pp")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_data(sizes, seed):
    rng = random.Random(seed)
    rows, provenance = {}, {}
    for (split, filename), size in zip(SPLITS.items(), sizes):
        source = (DATA / filename).read_bytes()
        examples = [json.loads(line) for line in source.splitlines() if line.strip()]
        indices = list(range(len(examples))) if size is None else rng.sample(range(len(examples)), size)
        rows[split] = [examples[index] for index in indices]
        provenance[split] = {
            "file": str((DATA / filename).relative_to(ROOT).as_posix()),
            "sha256": hashlib.sha256(source).hexdigest(),
            "pool_size": len(examples),
            "row_indices_zero_based_in_run_order": indices,
        }
    return rows, provenance


def read_call_stats(run_dir, validation_predictions, test_size):
    log_dir = run_dir / "detailed_llm_logs"
    records = [read_json(path) for path in log_dir.glob("*.json")]
    eval_calls = sum(record["call_id"].startswith("test_eval_") for record in records)
    if eval_calls != 2 * test_size + validation_predictions:
        raise RuntimeError(f"Incomplete evaluation in {run_dir}; only {eval_calls} successful evaluation calls.")
    if any(record.get("finish_reason") != "stop" for record in records):
        raise RuntimeError(f"A model reply was truncated or interrupted in {run_dir}.")
    truncated = [read_json(path) for path in log_dir.glob("truncated_attempts/*/*.json")]
    api_replies = records + truncated
    return {
        "api_calls": len(api_replies),
        "reflection_retry_calls": sum(
            record["role"] == "reflector" and "_round_" in record["call_id"] for record in records
        ),
        "input_tokens": sum(record["prompt_num_tokens"] for record in api_replies),
        "output_tokens": sum(record["response_num_tokens"] for record in api_replies),
        "served_models": sorted({record["served_model"] for record in api_replies}),
    }


def evaluate_base_llm(path, agent, processor, test_rows, settings):
    """Evaluate the dataset's original numeric-answer prompts without ACE."""
    from llm import timed_llm_call

    complete = path / "base_llm.json"
    if complete.exists():
        return read_json(complete)
    log_dir = path / f"base-llm-attempt-{uuid4().hex[:10]}"
    provider = settings["provider"].lower()

    def evaluate(index_and_row):
        index, row = index_and_row
        response, info = timed_llm_call(
            agent.generator_client, provider, settings["model"], row["context"],
            role="generator", call_id=f"base_llm_eval_{index}",
            max_tokens=settings["max_tokens"], log_dir=log_dir,
        )
        if info["finish_reason"] != "stop":
            raise RuntimeError(f"Base LLM test item {index} ended with {info['finish_reason']}.")
        prediction = response.strip()
        return {
            "index": index, "prediction": prediction, "target": row["target"],
            "correct": processor.answer_is_correct(prediction, row["target"]),
            "served_model": info["served_model"], "served_provider": info["served_provider"],
        }

    print(f"\nBASE LLM (original dataset prompts), {len(test_rows)} test items", flush=True)
    with ThreadPoolExecutor(max_workers=settings["workers"]) as pool:
        predictions = list(pool.map(evaluate, enumerate(test_rows)))
    correct = sum(item["correct"] for item in predictions)
    result = {"correct": correct, "total": len(test_rows), "predictions": predictions}
    write_json(complete, result)
    return result


def run_trial(path, condition, trial, rounds, rows, settings):
    from ace import ACE
    from eval.finance.data_processor import DataProcessor
    from playbook_utils import get_playbook_stats

    complete = path / "completed.json"
    if complete.exists():
        print(f"Reusing completed {condition} trial {trial}.", flush=True)
        return read_json(complete)

    attempt = path / f"attempt-{uuid4().hex[:10]}"
    processor = DataProcessor(task_name="formula")
    data = {split: processor.process_task_data(raw) for split, raw in rows.items()}
    provider = settings["provider"].lower()
    config = {
        "task_name": "formula",
        "num_epochs": settings["epochs"],
        "max_num_rounds": rounds,
        "curator_frequency": 1,
        "eval_steps": len(rows["train"]),
        "save_steps": len(rows["train"]),
        "json_mode": True,
        "no_ground_truth": False,
        "test_workers": settings["workers"],
        "use_bulletpoint_analyzer": False,
        "save_dir": str(attempt),
        "api_provider": provider,
    }
    print(f"\n{condition}, trial {trial}, reflection cap {rounds}", flush=True)
    agent = ACE(
        api_provider=provider,
        generator_model=settings["model"],
        reflector_model=settings["model"],
        curator_model=settings["model"],
        max_tokens=settings["max_tokens"],
    )
    try:
        base_llm = None
        if settings.get("base_llm_original_prompt") and condition == "baseline":
            base_llm = evaluate_base_llm(path, agent, processor, rows["test"], settings)
        start = time.perf_counter()
        result = agent.run(
            mode="offline", train_samples=data["train"], val_samples=data["val"],
            test_samples=data["test"], data_processor=processor, config=config,
        )
        elapsed = time.perf_counter() - start
    finally:
        for client in (agent.generator_client, agent.reflector_client, agent.curator_client):
            client.close()

    before = result["initial_test_results"]
    after = result["final_test_results"]
    expected_test_size = len(rows["test"])
    if before["total"] != expected_test_size or after["total"] != expected_test_size:
        raise RuntimeError(f"Incomplete test predictions for {condition} trial {trial}. See {attempt}.")
    run_dir, = attempt.glob("ace_run_*")
    checkpoints = read_json(run_dir / "train_results.json")["results"]
    if len(checkpoints) != settings["epochs"] or any(
        checkpoint["val_result"].get("total") != len(rows["val"]) for checkpoint in checkpoints
    ):
        raise RuntimeError(f"Incomplete validation checkpoints for {condition} trial {trial}. See {attempt}.")
    stats = read_call_stats(run_dir, settings["epochs"] * len(rows["val"]), expected_test_size)
    metrics = {
        "condition": condition, "trial": trial, "max_reflection_rounds": rounds,
        "test_size": expected_test_size, "empty_playbook_correct": before["correct"],
        "final_correct": after["correct"], "empty_playbook_accuracy_pct": 100 * before["accuracy"],
        "final_accuracy_pct": 100 * after["accuracy"],
        "accuracy_gain_pp": 100 * (after["accuracy"] - before["accuracy"]),
        "validation_accuracy_pct": 100 * result["training_results"]["best_validation_accuracy"],
        "wall_seconds": elapsed, **stats,
        "best_playbook_bullets": get_playbook_stats(agent.best_playbook)["total_bullets"],
        "curator_failures_logged": (run_dir / "detailed_llm_logs" / "curator_failures.txt").exists(),
        "run_directory": run_dir.relative_to(path.parent.parent).as_posix(),
    }
    metrics["served_models"] = ", ".join(metrics["served_models"])
    if settings.get("base_llm_original_prompt"):
        metrics.update(dict.fromkeys(BASE_LLM_METRICS, ""))
        if base_llm is not None:
            base_accuracy = base_llm["correct"] / base_llm["total"]
            metrics["base_llm_accuracy_pct"] = 100 * base_accuracy
            metrics["ace_minus_base_llm_pp"] = 100 * (after["accuracy"] - base_accuracy)
    write_json(complete, metrics)
    print(f"Validated {condition} trial {trial}. Saved {complete}.", flush=True)
    return metrics


def summarize(output, trial_rows):
    by_condition = {
        condition: [row for row in trial_rows if row["condition"] == condition]
        for condition in CONDITIONS
    }
    baseline = {row["trial"]: row for row in by_condition["baseline"]}
    shortened = {row["trial"]: row for row in by_condition["short_reflection"]}
    paired = [
        {"trial": trial, **{metric: shortened[trial][metric] - original[metric] for metric in METRICS}}
        for trial, original in baseline.items() if trial in shortened
    ]
    by_condition["short_reflection_minus_baseline"] = paired
    summary = []
    for condition, runs in by_condition.items():
        for metric in METRICS + BASE_LLM_METRICS:
            values = [row[metric] for row in runs if row.get(metric, "") != ""]
            if not values:
                continue
            summary.append({"comparison": condition, "metric": metric, "n": len(values),
                            "mean": statistics.mean(values),
                            "sample_variance": statistics.variance(values) if len(values) > 1 else ""})
    write_csv(output / "trials.csv", trial_rows)
    write_csv(output / "summary.csv", summary)
    if paired:
        write_csv(output / "paired_differences.csv", paired)


def verify_api(settings):
    """Check the configured route with one plain-text and one JSON request."""
    from llm import timed_llm_call
    from utils import initialize_clients

    provider = settings["provider"].lower()
    clients = initialize_clients(provider)
    try:
        for json_mode in (False, True):
            prompt = 'Reply only with the JSON object {"final_answer":"2"}.' if json_mode else 'Reply only with the number 2.'
            response, info = timed_llm_call(
                clients[0], provider, settings["model"], prompt, "generator",
                f"verify_{'json' if json_mode else 'text'}", max_tokens=128, use_json_mode=json_mode,
            )
            answer = json.loads(response)["final_answer"] if json_mode else response
            if info["finish_reason"] != "stop" or str(answer).strip() not in ("2", "2.0"):
                raise RuntimeError("API verification returned an unexpected answer or finish reason.")
            print(f"Verified {'JSON' if json_mode else 'plain text'}: "
                  f"model={info['served_model']}, provider={info['served_provider'] or provider}", flush=True)
    finally:
        for client in clients:
            client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", *PAPER_PROFILES), default="pilot")
    parser.add_argument("--phase", choices=("baseline", "ablation", "both"),
                        help="Ablation requires the saved baseline. paper-short defaults to both, baseline first.")
    parser.add_argument("--quick", action="store_true", help="One short smoke trial per condition.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan-only", action="store_true", help="Print the dataset and settings without using an API key.")
    action.add_argument("--verify-api", action="store_true", help="Check model routing, plain text and JSON with two small requests.")
    parser.add_argument("--trials", type=int, choices=range(1, 6),
                        help="Trials per condition (default: 1 for paper-short, 3 for other profiles).")
    parser.add_argument("--seed", type=int, default=5660, help="Seed for sampling data rows only.")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--provider", choices=("deepseek", "openrouter"), default="deepseek")
    parser.add_argument("--model", help="Defaults to deepseek-flash or the pinned OpenRouter DeepSeek V3.1 model.")
    parser.add_argument("--output", type=Path, help="Reuse the same path and arguments to resume incomplete work.")
    args = parser.parse_args()
    paper_profile = args.profile in PAPER_PROFILES
    short_profile = args.profile == "paper-short"
    if args.quick and paper_profile:
        parser.error("--quick uses the pilot profile. Use --verify-api to check a paper-profile endpoint.")
    phase = args.phase or ("baseline" if paper_profile and not short_profile else "both")
    trials_per_condition = 1 if args.quick else (args.trials or (1 if short_profile else 3))
    if paper_profile:
        sizes, epochs = PAPER_PROFILES[args.profile]
        output_name = f"formula-{args.profile}-{args.provider}"
    else:
        sizes = (2, 2, 3) if args.quick else (12, 8, 30)
        epochs = 1
        output_name = "formula-smoke" if args.quick else "formula-study"
    output = (args.output or ROOT / "results" / output_name).resolve()
    rows, provenance = select_data(sizes, args.seed)
    # Read project .env from its known location. An exported variable takes precedence.
    from dotenv import load_dotenv
    from llm import DEEPSEEK_TOP_P, REQUEST_TEMPERATURE
    from utils import OPENROUTER_BASE_URL, OPENROUTER_V31_ENDPOINT, OPENROUTER_V31_MODEL, deepseek_base_url

    load_dotenv(ROOT / ".env")
    openrouter = args.provider == "openrouter"
    if openrouter and args.model not in (None, OPENROUTER_V31_MODEL):
        parser.error(f"The OpenRouter route is pinned for {OPENROUTER_V31_MODEL}.")
    model = args.model or (OPENROUTER_V31_MODEL if openrouter else "deepseek-flash")
    settings = {
        "upstream_commit": UPSTREAM_COMMIT, "provider": "OpenRouter" if openrouter else "DeepSeek",
        "base_url": OPENROUTER_BASE_URL if openrouter else deepseek_base_url(),
        "model": model, "thinking": "disabled", "temperature": REQUEST_TEMPERATURE,
        "top_p": DEEPSEEK_TOP_P, "max_tokens": 4096, "json_mode": True, "prompt_changes": "none",
        "mode": "offline", "epochs": epochs,
        "batch_size": 1, "train_labels_for_reflection": True,
        "curator_frequency": 1, "validation_interval": len(rows["train"]), "bulletpoint_analyzer": False,
        "workers": args.workers, "data_seed": args.seed, "trials_per_condition": trials_per_condition,
        "condition_max_reflection_rounds": CONDITIONS, "dataset": provenance,
        "paper_table_2_formula_pct": {"base_llm": 67.5, "offline_ace_with_labels": 85.5},
    }
    if openrouter:
        settings["provider_endpoint"] = OPENROUTER_V31_ENDPOINT
    if paper_profile:
        settings.update(profile=args.profile, base_llm_original_prompt=True)
    if args.plan_only:
        print(json.dumps(settings, indent=2))
        return
    key_name = "OPENROUTER_API_KEY" if openrouter else "DEEPSEEK_API_KEY"
    if not os.getenv(key_name):
        parser.error(f"Set {key_name} in .env or in your shell before running the API experiment.")
    if args.verify_api:
        verify_api(settings)
        return
    manifest = output / "study_config.json"
    if manifest.exists() and read_json(manifest) != settings:
        parser.error(f"Settings differ from {manifest}. Use a new --output directory.")
    trials = range(1, settings["trials_per_condition"] + 1)
    completed_paths = {
        (trial, condition): output / condition / f"trial-{trial:02d}" / "completed.json"
        for trial in trials for condition in CONDITIONS
    }
    if phase == "ablation" and (not manifest.is_file() or not all(
        completed_paths[trial, "baseline"].is_file() for trial in trials
    )):
        parser.error("Complete --phase baseline with these settings and this output directory first.")
    write_json(manifest, settings)

    if phase == "baseline":
        schedule = [(trial, "baseline") for trial in trials]
    elif phase == "ablation":
        schedule = [(trial, "short_reflection") for trial in trials]
    elif paper_profile:
        schedule = [(trial, condition) for condition in CONDITIONS for trial in trials]
    else:
        schedule = [
            (trial, condition) for trial in trials
            for condition in (CONDITIONS if trial % 2 else reversed(CONDITIONS))
        ]
    for trial, condition in schedule:
        run_trial(completed_paths[trial, condition].parent, condition, trial, CONDITIONS[condition], rows, settings)
        summarize(output, [read_json(path) for path in completed_paths.values() if path.is_file()])
    print(f"Results: {output / 'summary.csv'} (see also trials.csv)", flush=True)


if __name__ == "__main__":
    main()
