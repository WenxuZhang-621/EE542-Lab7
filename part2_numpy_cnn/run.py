"""Train and evaluate Part II. Run from the repository root with Python."""
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import PIL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from part2_numpy_cnn.data import assert_disjoint, load_split
from part2_numpy_cnn.model import NumpyCNN


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def cpu_name():
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def train(model, split, epochs, lr, feature_batch_size, seed):
    rng = np.random.default_rng(seed)
    history = []
    train_seconds = 0.0
    for epoch in range(1, epochs + 1):
        start = perf_counter()
        order = rng.permutation(len(split.labels))
        loss_sum = 0.0
        for offset in range(0, len(order), feature_batch_size):
            ids = order[offset:offset + feature_batch_size]
            # Recompute CNN features EVERY epoch. No persistent feature cache.
            features = model.features(split.images[ids])
            for flat, label in zip(features, split.labels[ids]):
                loss_sum += model.train_one(flat, label, lr)
        elapsed = perf_counter() - start
        loss = loss_sum / len(order)
        if not np.isfinite(loss) or not np.isfinite(model.W_fc).all():
            raise ValueError("Training diverged; reduce --learning-rate")
        train_seconds += elapsed
        history.append({"epoch": epoch, "online_mse": loss, "seconds": elapsed})
        print(f"Epoch {epoch:2d}/{epochs}: online MSE={loss:.6f}, {elapsed:.3f}s", flush=True)
    return history, train_seconds


def evaluate(model, split, block_size, threshold):
    scores = np.empty(len(split.labels), dtype=np.float32)
    blocks = []
    test_seconds = 0.0
    for offset in range(0, len(scores), block_size):
        end = min(offset + block_size, len(scores))
        start = perf_counter()
        block_scores = model.predict(split.images[offset:end])
        predicted = (block_scores > threshold).astype(np.int64)
        truth = split.labels[offset:end]
        correct = int(np.count_nonzero(predicted == truth))
        mse = float(np.mean((block_scores - truth) ** 2))
        elapsed = perf_counter() - start
        if not np.isfinite(block_scores).all():
            raise ValueError("Non-finite test predictions")
        test_seconds += elapsed
        scores[offset:end] = block_scores
        blocks.append({"block": len(blocks) + 1, "start": offset, "stop": end,
                       "num_images": end - offset, "correct": correct,
                       "accuracy": correct / (end - offset), "mse": mse,
                       "seconds": elapsed, "images_per_second": (end - offset) / elapsed})
        print(f"Test block {len(blocks)}: {end-offset} images, {elapsed:.3f}s", flush=True)
    predicted = (scores > threshold).astype(np.int64)
    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (split.labels, predicted), 1)
    metrics = {
        "accuracy": float(np.mean(predicted == split.labels)),
        "mse": float(np.mean((scores - split.labels) ** 2)),
        "correct": int(np.count_nonzero(predicted == split.labels)),
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_axes": "rows=true, columns=predicted; cat=0, dog=1",
        "per_class_accuracy": {cls: float(confusion[i, i] / confusion[i].sum())
                               for i, cls in enumerate(config.CLASSES)},
        "zero_output_fraction": float(np.mean(scores == 0.0)),
    }
    return metrics, blocks, scores, test_seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--output", default="results/part2_numpy.json")
    parser.add_argument("--epochs", type=positive_int, default=config.EPOCHS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--learning-rate", type=positive_float, default=0.001)
    parser.add_argument("--feature-batch-size", type=positive_int, default=64)
    parser.add_argument("--eval-block-size", type=positive_int, default=256)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing result bundle")
    args = parser.parse_args(argv)
    if config.CLASSES != ["cat", "dog"]:
        parser.error("Expected shared labels cat=0, dog=1")
    output = resolve(args.output)
    checkpoint = output.with_name(output.stem + "_model.npz")
    predictions_file = output.with_name(output.stem + "_predictions.csv")
    if not args.overwrite and any(p.exists() for p in [output, checkpoint, predictions_file]):
        parser.error("Result files already exist; choose --output or explicitly use --overwrite")
    data_dir = resolve(args.data_dir)
    overall_start = perf_counter()
    start = perf_counter()
    train_split = load_split(data_dir / "train", config.CLASSES, config.IMG_SIZE)
    train_load_seconds = perf_counter() - start
    start = perf_counter()
    test_split = load_split(data_dir / "test", config.CLASSES, config.IMG_SIZE)
    test_load_seconds = perf_counter() - start
    assert_disjoint(train_split, test_split)
    print(f"Loaded {len(train_split.labels)} train and {len(test_split.labels)} unseen test images", flush=True)
    model = NumpyCNN(config.IMG_SIZE, args.seed)
    history, train_seconds = train(model, train_split, args.epochs, args.learning_rate,
                                   args.feature_batch_size, args.seed)
    metrics, blocks, scores, test_seconds = evaluate(model, test_split, args.eval_block_size,
                                                    config.THRESHOLD)
    pipeline_seconds = perf_counter() - overall_start
    warnings = []
    if metrics["zero_output_fraction"] == 1.0:
        warnings.append("All outputs are zero: the output ReLU is inactive; results are not evidence of learning.")
    if min(np.sum(metrics["confusion_matrix"], axis=0)) == 0:
        warnings.append("Only one class was predicted on the test set.")
    result = {
        "schema_version": 1,
        "part": "part2_numpy",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "framework": "numpy", "device": "cpu",
        "dataset": data_dir.name,
        "num_train": len(train_split.labels), "num_test": len(test_split.labels),
        "epochs": args.epochs, "seed": args.seed,
        "accuracy": metrics["accuracy"],
        "train_seconds": train_seconds, "test_seconds": test_seconds,
        "test_images_per_second": len(test_split.labels) / test_seconds,
        "settings": {"image_size": config.IMG_SIZE, "classes": config.CLASSES,
                     "threshold": config.THRESHOLD, "learning_rate": args.learning_rate,
                     "optimizer": "online SGD", "optimizer_batch_size": 1,
                     "feature_batch_size": args.feature_batch_size,
                     "eval_block_size": args.eval_block_size, "dtype": "float32",
                     "preprocessing": "Pillow RGB, BILINEAR resize, /255.0; NHWC",
                     "loss": "MSE", "feature_cache": False},
        "architecture": {"conv_filter_shape": [3, 3, 3], "conv_padding": "valid",
                         "conv_output_shape": [config.IMG_SIZE-2, config.IMG_SIZE-2],
                         "pool_size": 2, "flatten_dim": model.feature_dim,
                         "fc_output_dim": 1, "output_activation": "ReLU",
                         "trainable_parameters": int(model.W_fc.size + model.b_fc.size),
                         "fixed_parameters": int(model.conv_filter.size)},
        "data": {"counts": {"train": train_split.counts, "test": test_split.counts},
                 "fingerprints_sha256": {"train": train_split.fingerprint, "test": test_split.fingerprint},
                 "fingerprint_format": "SHA256 of sorted class/filename records: label<TAB>relative_path<TAB>file_sha256<LF>",
                 "overlap_check": "No shared relative filenames or identical file bytes"},
        "timing": {"train_load_preprocess_seconds": train_load_seconds,
                   "test_load_preprocess_seconds": test_load_seconds,
                   "load_preprocess_seconds": train_load_seconds + test_load_seconds,
                   "train_seconds": train_seconds, "test_seconds": test_seconds,
                   "pipeline_seconds_before_export": pipeline_seconds,
                   "train_images_per_second": len(train_split.labels) * args.epochs / train_seconds,
                   "test_images_per_second": len(test_split.labels) / test_seconds,
                   "definition": "perf_counter; train=sum epoch compute timers (shuffle, CNN, online SGD); test=sum block timers (CNN, threshold, metrics); loading, hash checks, console output and file export excluded from compute timers. No feature cache. One run, no explicit warm-up."},
        "metrics": metrics, "history": history, "test_blocks": blocks,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pillow": PIL.__version__, "platform": platform.platform(),
                        "cpu": cpu_name(), "logical_cpus": os.cpu_count(),
                        "thread_environment": {k: os.environ.get(k) for k in
                            ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}},
        "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ["run.py", "model.py", "data.py"]},
        "warnings": warnings,
        "comparison_note": "Part II uses one fixed random filter and trains only FC. Part III has two trainable convolution layers; speed/accuracy differences are not solely GPU effects. Match data, resize interpolation and timing boundaries.",
        "artifacts": {"model": checkpoint.name, "predictions": predictions_file.name},
    }
    manifest = data_dir / "manifest.json"
    if manifest.exists():
        result["data"]["preparation_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    with predictions_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "true_label", "predicted_label", "score"])
        for name, label, score in zip(test_split.files, test_split.labels, scores):
            writer.writerow([name, int(label), int(score > config.THRESHOLD), float(score)])
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Test accuracy: {100*metrics['accuracy']:.2f}% | train: {train_seconds:.3f}s | test: {test_seconds:.3f}s")
    print(f"Saved {output}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return result


if __name__ == "__main__":
    main()
