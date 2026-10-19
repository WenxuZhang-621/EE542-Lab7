"""Repeat seed 542 twice, then inspect predeclared seeds 543, 544 and 545.

All timed runs execute sequentially in fresh Python processes. The data split
and model source remain unchanged. The original baseline is never overwritten.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from part2_numpy_cnn.data import load_split
from part2_numpy_cnn.model import NumpyCNN


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="results/part2_numpy.json")
    parser.add_argument("--runs-dir", default="results/part2_numpy_repeats")
    parser.add_argument("--summary", default="results/part2_numpy_summary.json")
    args = parser.parse_args()
    baseline_path = ROOT / args.baseline
    baseline_bytes = baseline_path.read_bytes()
    baseline = read(baseline_path)
    if baseline["seed"] != 542:
        raise ValueError("This protocol expects the original seed-542 baseline")
    for name, digest in baseline["source_sha256"].items():
        if hashlib.sha256((ROOT/"part2_numpy_cnn"/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Baseline source changed: {name}; repeat comparisons would be invalid")
    runs_dir = ROOT / args.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    experiments = [("seed542_run2", 542, "timing_repeat"),
                   ("seed542_run3", 542, "timing_repeat"),
                   ("seed543", 543, "seed_sensitivity"),
                   ("seed544", 544, "seed_sensitivity"),
                   ("seed545", 545, "seed_sensitivity")]
    protocol = {
        "baseline": args.baseline,
        "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "timing_group": "Original seed 542 run plus two same-seed repeats",
        "sensitivity_group": "Seeds 542, 543, 544, 545; count seed 542 only once",
        "new_experiments": [{"name": name, "seed": seed, "group": group}
                            for name, seed, group in experiments],
        "selection_policy": "Additional seeds declared before running them. Keep every result; no best-seed selection.",
        "fixed": {k: baseline[k] for k in ["dataset", "num_train", "num_test", "epochs", "settings", "architecture", "source_sha256"]},
        "data_split_seed": baseline["data"]["preparation_manifest"]["seed"],
        "execution": "Sequential fresh Python processes; model initialized and trained from scratch in each run. No explicit warm-up or OS filesystem cache control. Original run occurred earlier in this session.",
    }
    plan_path = runs_dir / "protocol.json"
    if plan_path.exists():
        if read(plan_path) != protocol:
            raise ValueError("Existing experiment protocol differs; use a new --runs-dir")
    else:
        # Write before starting the first supplementary experiment.
        plan_path.write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8")
    rows = [("seed542_run1", "timing_repeat", baseline_path, baseline)]
    for name, seed, group in experiments:
        output = runs_dir / (name + ".json")
        if not output.exists():
            command = [sys.executable, str(ROOT/"part2_numpy_cnn/run.py"),
                       "--data-dir", baseline["dataset"], "--output", str(output),
                       "--seed", str(seed), "--epochs", str(baseline["epochs"]),
                       "--learning-rate", str(baseline["settings"]["learning_rate"]),
                       "--feature-batch-size", str(baseline["settings"]["feature_batch_size"]),
                       "--eval-block-size", str(baseline["settings"]["eval_block_size"])]
            print(f"Running {name}: seed {seed}, {baseline['epochs']} epochs", flush=True)
            with (runs_dir / (name+".log")).open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = read(output)
        for key in ["dataset", "num_train", "num_test", "epochs", "settings", "architecture", "source_sha256", "environment"]:
            if result[key] != baseline[key]:
                raise ValueError(f"{name}: {key} differs from baseline")
        if result["seed"] != seed or result["data"] != baseline["data"]:
            raise ValueError(f"{name}: seed or dataset mismatch")
        print(f"{name}: accuracy={100*result['accuracy']:.2f}%, train={result['train_seconds']:.6f}s, test={result['test_seconds']:.6f}s", flush=True)
        rows.append((name, group, output, result))
    if baseline_path.read_bytes() != baseline_bytes:
        raise ValueError("Baseline was modified")

    # Diagnostics run AFTER all timed experiments. Only training images are
    # used to characterize the random feature map and output-ReLU activity.
    train = load_split(ROOT/baseline["dataset"]/"train", baseline["settings"]["classes"],
                       baseline["settings"]["image_size"])
    diagnostics = {}
    for name, group, path, result in rows:
        seed = result["seed"]
        if seed in diagnostics:
            continue
        initial = NumpyCNN(baseline["settings"]["image_size"], seed)
        final = NumpyCNN(baseline["settings"]["image_size"], seed)
        with np.load(path.parent/result["artifacts"]["model"]) as saved:
            for field in ["conv_filter", "W_fc", "b_fc"]:
                setattr(final, field, saved[field])
        nonzero = total = initial_zero = final_zero = 0
        final_scores = []
        for offset in range(0, len(train.labels), 64):
            features = initial.features(train.images[offset:offset+64])
            nonzero += int(np.count_nonzero(features))
            total += features.size
            initial_zero += int(np.count_nonzero(initial.predict_features(features) == 0))
            predictions = final.predict_features(features)
            final_zero += int(np.count_nonzero(predictions == 0))
            final_scores.append(predictions)
        scores = np.concatenate(final_scores)
        diagnostics[seed] = {
            "training_pooled_nonzero_fraction": nonzero/total,
            "initial_training_zero_output_fraction": initial_zero/len(train.labels),
            "final_training_zero_output_fraction": final_zero/len(train.labels),
            "final_training_accuracy": float(np.mean((scores > result["settings"]["threshold"]) == train.labels)),
            "final_training_score_min": float(scores.min()),
            "final_training_score_max": float(scores.max()),
            "fixed_kernel_sum": float(initial.conv_filter.sum()),
        }

    original_predictions = (baseline_path.parent/baseline["artifacts"]["predictions"]).read_bytes()
    compact = []
    for name, group, path, result in rows:
        prediction_file = path.parent/result["artifacts"]["predictions"]
        cm = np.asarray(result["metrics"]["confusion_matrix"])
        compact.append({
            "run_id": name, "group": group, "seed": result["seed"],
            "result_file": path.relative_to(ROOT).as_posix(),
            "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "accuracy": result["accuracy"], "mse": result["metrics"]["mse"],
            "train_seconds": result["train_seconds"], "test_seconds": result["test_seconds"],
            "load_preprocess_seconds": result["timing"]["load_preprocess_seconds"],
            "test_images_per_second": result["test_images_per_second"],
            "predicted_cat": int(cm[:, 0].sum()), "predicted_dog": int(cm[:, 1].sum()),
            "confusion_matrix": cm.tolist(),
            "zero_output_fraction": result["metrics"]["zero_output_fraction"],
            "identical_prediction_csv_to_baseline": prediction_file.read_bytes() == original_predictions,
            "training_diagnostics": diagnostics[result["seed"]],
        })
    timing_rows = [r for r in compact if r["group"] == "timing_repeat"]
    seed_rows = [compact[0]] + [r for r in compact if r["group"] == "seed_sensitivity"]
    summary = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol, "total_complete_runs": len(compact),
        "new_complete_runs": len(experiments), "runs": compact,
        "timing_summary": {key: stats([r[key] for r in timing_rows]) for key in
                           ["train_seconds", "test_seconds", "load_preprocess_seconds", "test_images_per_second"]},
        "seed_sensitivity_summary": {
            "unique_seeds": [r["seed"] for r in seed_rows],
            "accuracy": stats([r["accuracy"] for r in seed_rows]),
            "mse": stats([r["mse"] for r in seed_rows]),
        },
        "std_definition": "Sample standard deviation (ddof=1). Timing statistics use only three seed-542 runs; seed sensitivity counts each of four seeds once.",
        "data_fingerprints_sha256": baseline["data"]["fingerprints_sha256"],
        "diagnostics_note": "Measured on all 2000 training images after timed runs; not included in training/test compute timings. Changing --seed changes fixed filter, FC initialization and training shuffle together; this experiment cannot separate their effects.",
        "limitations": ["Same held-out images in every run: no additional independent test images.",
                        "Three timing samples and four seeds are small samples, not proof of general robustness.",
                        "No hardware isolation or OS cache control; original baseline run predates the repeats.",
                        "Architecture and decision threshold unchanged; all outcomes retained, no test-based model selection."],
    }
    summary_path = ROOT / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    fields = ["run_id", "group", "seed", "accuracy", "mse", "load_preprocess_seconds",
              "train_seconds", "test_seconds", "test_images_per_second", "predicted_cat", "predicted_dog"]
    with summary_path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(compact)
    print(json.dumps({"timing_summary": summary["timing_summary"],
                      "seed_sensitivity_summary": summary["seed_sensitivity_summary"]}, indent=2), flush=True)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
