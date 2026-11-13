"""Recover the final Formula test of an interrupted paper-subset baseline."""
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ace.core.generator import Generator
from ace.prompts.generator import GENERATOR_PROMPT
from eval.finance.data_processor import DataProcessor
from experiments.formula import read_json, summarize, write_json
from utils import extract_answer, initialize_clients

OUTPUT = Path("results/formula-paper-subset-deepseek")
TRIAL = OUTPUT / "baseline" / "trial-01"
RECOVERY = OUTPUT / "recovery-trial-01"


def load_test(manifest):
    spec = manifest["dataset"]["test"]
    data = Path(spec["file"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError("The test file differs from the original run's dataset.")
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    return [rows[index] for index in spec["row_indices_zero_based_in_run_order"]]


def saved_answers(log_dir, prompts, after=None):
    answers = {}
    for path in sorted(log_dir.glob("generator_test_eval_*.json")):
        if after is not None and path.stat().st_mtime_ns < after:
            continue
        try:
            record = read_json(path)
        except (ValueError, OSError):
            continue
        match = re.fullmatch(r"test_eval_(\d+)", record.get("call_id", ""))
        if match is None or record.get("finish_reason") != "stop":
            continue
        index = int(match[1])
        response = record.get("response")
        if index < len(prompts) and record.get("prompt") == prompts[index] and isinstance(response, str):
            answers[index] = extract_answer(response)
    return answers


def main():
    manifest = read_json(OUTPUT / "study_config.json")
    expected = ("paper-subset", "deepseek", 5)
    actual = (manifest["profile"], manifest["provider"].lower(), manifest["epochs"])
    if actual != expected or not manifest.get("base_llm_original_prompt"):
        raise ValueError("The manifest does not match the DeepSeek paper-subset run.")
    test_rows = load_test(manifest)
    processor = DataProcessor(task_name="formula")
    samples = processor.process_task_data(test_rows)
    total = len(samples)
    if total != 200:
        raise ValueError(f"Expected 200 test rows, found {total}.")

    checkpoints = list(TRIAL.glob("attempt-*/ace_run_*/best_playbook.txt"))
    if not checkpoints:
        raise FileNotFoundError("No saved best_playbook.txt found for baseline trial 1.")
    best = max(checkpoints, key=lambda path: path.stat().st_mtime_ns)
    run = best.parent
    original_final = run / "final_test_results.json"
    train = read_json(run / "train_results.json")
    validation_total = len(manifest["dataset"]["val"]["row_indices_zero_based_in_run_order"])
    if len(train["results"]) != manifest["epochs"] or any(
        step["val_result"]["total"] != validation_total for step in train["results"]
    ):
        raise ValueError("The selected run has incomplete validation checkpoints.")

    base = read_json(TRIAL / "base_llm.json")
    initial = read_json(run / "initial_test_results.json")["test_results"]
    if base["total"] != total or initial["total"] != total:
        raise ValueError("The base or initial test does not cover the full test set.")

    if original_final.is_file():
        recorded = read_json(original_final)
        if recorded["test_results"]["total"] == total:
            final = recorded["test_results"]
            audit = {"method": "original_final_test_results", "source_run": str(run)}
        else:
            final = None
    else:
        final = None

    if final is None:
        playbook = best.read_text()
        prompts = [
            GENERATOR_PROMPT.format(playbook, "(empty)", sample["question"], sample["context"])
            for sample in samples
        ]
        logs = run / "detailed_llm_logs"
        original = saved_answers(logs, prompts, after=best.stat().st_mtime_ns)
        recovery_logs = RECOVERY / "logs"
        recovered = saved_answers(recovery_logs, prompts)
        answers = {**recovered, **original}
        missing = sorted(set(range(total)) - answers.keys())
        print(f"Original final-test answers: {len(original)}/{total}. API calls needed now: {len(missing)}.", flush=True)
        if missing:
            clients = initialize_clients(actual[1])
            generator = Generator(clients[0], actual[1], manifest["model"], manifest["max_tokens"])

            def evaluate(index):
                sample = samples[index]
                response, _, info = generator.generate(
                    question=sample["question"], playbook=playbook, context=sample["context"],
                    use_json_mode=manifest["json_mode"], call_id=f"test_eval_{index}", log_dir=recovery_logs,
                )
                if info["finish_reason"] != "stop":
                    raise RuntimeError(f"Test item {index} returned an incomplete response.")
                return index, extract_answer(response)

            try:
                with ThreadPoolExecutor(max_workers=manifest["workers"]) as pool:
                    answers.update(pool.map(evaluate, missing))
            finally:
                for client in clients:
                    client.close()
        targets = [sample["target"] for sample in samples]
        ordered = [answers[index] for index in range(total)]
        accuracy = processor.evaluate_accuracy(ordered, targets)
        final = {"correct": round(accuracy * total), "total": total, "accuracy": accuracy}
        audit = {
            "method": "recover_saved_answers_and_fill_missing",
            "source_run": str(run),
            "original_answers": len(original),
            "recovery_items": sorted(set(range(total)) - original.keys()),
        }

    base_pct = 100 * base["correct"] / total
    initial_pct = 100 * initial["correct"] / total
    final_pct = 100 * final["correct"] / total
    row = {
        "condition": "baseline",
        "trial": 1,
        "max_reflection_rounds": manifest["condition_max_reflection_rounds"]["baseline"],
        "test_size": total,
        "base_llm_correct": base["correct"],
        "empty_playbook_correct": initial["correct"],
        "final_correct": final["correct"],
        "base_llm_accuracy_pct": base_pct,
        "empty_playbook_accuracy_pct": initial_pct,
        "final_accuracy_pct": final_pct,
        "accuracy_gain_pp": final_pct - initial_pct,
        "ace_minus_base_llm_pp": final_pct - base_pct,
        "validation_accuracy_pct": 100 * train["best_accuracy"],
    }
    write_json(RECOVERY / "final_test_results.json", {"test_results": final, "recovery": audit})
    summarize(RECOVERY, [row])
    print(f"Base model: {base_pct:.2f}% | Final ACE: {final_pct:.2f}% | Change: {final_pct - base_pct:+.2f} pp")
    print(f"Summary: {RECOVERY / 'summary.csv'}")


if __name__ == "__main__":
    main()
