"""Part III: train and evaluate the CNN with PyTorch on the GPU.

Run from the repository root:
    python part3_torch_cnn/run.py                       # handout model, GPU
    python part3_torch_cnn/run.py --help                # all options

Timing follows Part II: training and test timers cover computation only;
loading and decoding images is timed separately. On a GPU every timer
synchronizes the device first and last, because kernels run asynchronously
and an unsynchronized timer only measures how long it took to queue them.
"""
import argparse
from datetime import datetime, timezone
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from part3_torch_cnn.data import assert_disjoint, image_folder, load_split  # noqa: E402
from part3_torch_cnn.model import build, describe  # noqa: E402


# ---------------------------------------------------------------- utilities

def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def clock(device):
    """Wall-clock time after all queued device work has finished."""
    sync(device)
    return perf_counter()


def cpu_name():
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def gpu_driver():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None


def environment(device):
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "torchvision": torchvision.__version__, "numpy": np.__version__,
           "platform": platform.platform(), "cpu": cpu_name(),
           "logical_cpus": os.cpu_count(), "torch_cpu_threads": torch.get_num_threads(),
           "device": str(device)}
    if device.type == "cuda":
        env.update({"gpu": torch.cuda.get_device_name(device),
                    "gpu_capability": "sm_%d%d" % torch.cuda.get_device_capability(device),
                    "cuda_runtime": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "nvidia_driver": gpu_driver()})
    return env


# ---------------------------------------------------------------- training

def make_loss(head):
    return nn.MSELoss() if head == "relu_mse" else nn.BCEWithLogitsLoss()


def make_optimizer(name, params, lr):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(name)


def train_batches(model, opt, loss_fn, X, Y, batch_size, epochs, seed, device, max_batches=None):
    """Mini-batch training on tensors already in device memory. The shuffle
    order comes from a seeded CPU generator, so it is the same on any device."""
    gen = torch.Generator().manual_seed(seed)
    history, total = [], 0.0
    n = len(Y)
    for epoch in range(1, epochs + 1):
        start = clock(device)
        order = torch.randperm(n, generator=gen).to(device)
        loss_sum = torch.zeros((), device=device)
        for b, off in enumerate(range(0, n, batch_size)):
            if max_batches is not None and b >= max_batches:
                break
            idx = order[off:off + batch_size]
            x, y = X[idx], Y[idx]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * len(idx)   # stays on the device: no sync per batch
        elapsed = clock(device) - start
        if max_batches is not None:
            return [], elapsed
        total += elapsed
        mean_loss = float(loss_sum) / n
        if not np.isfinite(mean_loss):
            raise ValueError("training diverged")
        history.append({"epoch": epoch, "loss": mean_loss, "seconds": elapsed})
        print(f"Epoch {epoch:2d}/{epochs}: loss={mean_loss:.6f}, {elapsed:.3f}s", flush=True)
    return history, total


def train_online_numpy(model, X, Y, epochs, lr, seed, feature_batch, device, max_samples=None):
    """Part II's algorithm step for step: one SGD update per image, in the
    order NumPy's default_rng(seed).permutation gives, with features computed
    in groups of feature_batch and recomputed every epoch."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.SGD(model.fc.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history, total = [], 0.0
    n = len(Y)
    for epoch in range(1, epochs + 1):
        start = clock(device)
        order = rng.permutation(n)
        loss_sum = torch.zeros((), device=device)
        done = 0
        for off in range(0, n, feature_batch):
            ids = torch.from_numpy(order[off:off + feature_batch]).to(device)
            with torch.no_grad():
                feats = model.features(X[ids])
            labels = Y[ids]
            for i in range(len(ids)):
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model.forward_features(feats[i:i + 1]), labels[i:i + 1])
                loss.backward()
                opt.step()
                loss_sum += loss.detach()
                done += 1
                if max_samples is not None and done >= max_samples:
                    return [], clock(device) - start
        elapsed = clock(device) - start
        total += elapsed
        mean_loss = float(loss_sum) / n
        history.append({"epoch": epoch, "online_mse": mean_loss, "seconds": elapsed})
        print(f"Epoch {epoch:2d}/{epochs}: online MSE={mean_loss:.6f}, {elapsed:.3f}s", flush=True)
    return history, total


def train_from_disk(model, opt, loss_fn, split_dir, batch_size, epochs, seed, device, num_workers):
    """The handout's input pipeline: ImageFolder + DataLoader every epoch, so
    each epoch also decodes and resizes every image on the CPU. Used to show
    how much of the time the input pipeline takes."""
    ds = image_folder(split_dir, config.CLASSES, config.IMG_SIZE)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen,
                        num_workers=num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=num_workers > 0)
    history, total = [], 0.0
    for epoch in range(1, epochs + 1):
        start = clock(device)
        loss_sum = torch.zeros((), device=device)
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * len(y)
        elapsed = clock(device) - start
        total += elapsed
        mean_loss = float(loss_sum) / len(ds)
        history.append({"epoch": epoch, "loss": mean_loss, "seconds": elapsed})
        print(f"Epoch {epoch:2d}/{epochs}: loss={mean_loss:.6f}, {elapsed:.3f}s (incl. decoding)", flush=True)
    return history, total


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model, X, Y, block_size, threshold, device):
    """Classify the unseen test images in blocks, timing each block from the
    first kernel to the computed accuracy (as Part II does)."""
    model.eval()
    model(X[:block_size])                     # warm-up: kernel selection, lazy init
    sync(device)
    scores = torch.empty(len(Y), device=device)
    blocks, total = [], 0.0
    for off in range(0, len(Y), block_size):
        end = min(off + block_size, len(Y))
        start = clock(device)
        s = model.score(model(X[off:end]))
        correct = int(((s > threshold).long() == Y[off:end]).sum())   # .item() waits for the GPU
        elapsed = clock(device) - start
        total += elapsed
        scores[off:end] = s
        blocks.append({"block": len(blocks) + 1, "start": off, "stop": end,
                       "num_images": end - off, "correct": correct,
                       "accuracy": correct / (end - off), "seconds": elapsed,
                       "images_per_second": (end - off) / elapsed})
    scores = scores.cpu()
    labels = Y.cpu()
    pred = (scores > threshold).long()
    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (labels.numpy(), pred.numpy()), 1)
    metrics = {
        "accuracy": float((pred == labels).float().mean()),
        "correct": int((pred == labels).sum()),
        "mse": float(((scores - labels.float()) ** 2).mean()),
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_axes": "rows=true, columns=predicted; cat=0, dog=1",
        "per_class_accuracy": {c: float(confusion[i, i] / confusion[i].sum())
                               for i, c in enumerate(config.CLASSES)},
        "zero_output_fraction": float((scores == 0).float().mean()),
    }
    return metrics, blocks, scores, total


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=["catdog", "numpy_equiv"], default="catdog",
                    help="catdog: the handout's CatDogCNN; numpy_equiv: Part II's network in PyTorch")
    ap.add_argument("--head", choices=["relu_mse", "bce"], default="relu_mse",
                    help="relu_mse: handout (ReLU output, MSE); bce: logit output, BCEWithLogitsLoss")
    ap.add_argument("--train-mode", choices=["batch", "online"], default="batch",
                    help="online: Part II's one-update-per-image SGD (numpy_equiv only)")
    ap.add_argument("--loader", choices=["preload", "disk"], default="preload",
                    help="preload: decode once, keep tensors on the device; disk: ImageFolder + DataLoader every epoch")
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--data-dir", default=config.DATA_DIR)
    ap.add_argument("--device", default="auto", help="auto (cuda if available), cuda, cpu, mps")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for --loader disk")
    ap.add_argument("--eval-block-size", type=int, default=256)
    ap.add_argument("--output", default="results/part3_torch.json")
    ap.add_argument("--label", default=None, help="short name of this run for the comparison table")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    if config.CLASSES != ["cat", "dog"]:
        ap.error("expected shared labels cat=0, dog=1")
    if args.train_mode == "online" and args.model != "numpy_equiv":
        ap.error("--train-mode online reproduces Part II and needs --model numpy_equiv")
    if args.train_mode == "online" and args.loader != "preload":
        ap.error("--train-mode online uses preloaded tensors")
    output = resolve(args.output)
    weights = output.with_name(output.stem + "_model.pt")
    predictions = output.with_name(output.stem + "_predictions.csv")
    if not args.overwrite and any(p.exists() for p in (output, weights, predictions)):
        ap.error(f"{output.name} or its artifacts exist; choose --output or use --overwrite")

    device = pick_device(args.device)
    print("Using device:", device, f"({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
    data_dir = resolve(args.data_dir)
    overall = perf_counter()

    t = perf_counter()
    train_split = load_split(data_dir / "train", config.CLASSES, config.IMG_SIZE)
    train_load = perf_counter() - t
    t = perf_counter()
    test_split = load_split(data_dir / "test", config.CLASSES, config.IMG_SIZE)
    test_load = perf_counter() - t
    assert_disjoint(train_split, test_split)
    print(f"Loaded {len(train_split.labels)} train and {len(test_split.labels)} unseen test images; "
          f"one batch: {tuple(train_split.images[:args.batch_size].shape)}", flush=True)

    t = clock(device)
    X_train = train_split.images.to(device)
    Y_train = train_split.labels.to(device).float()
    X_test = test_split.images.to(device)
    Y_test = test_split.labels.to(device)
    h2d = clock(device) - t

    model = build(args.model, config.IMG_SIZE, args.seed, args.head).to(device)
    loss_fn = make_loss(model.head)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Warm-up on a throwaway copy: the first kernels pay for loading code and
    # choosing algorithms, which a one-off measurement should not include.
    warm = copy.deepcopy(model)
    if args.train_mode == "online":
        train_online_numpy(warm, X_train, Y_train, 1, args.lr, args.seed, 64, device, max_samples=64)
    else:
        train_batches(warm, make_optimizer(args.optimizer, [p for p in warm.parameters() if p.requires_grad], args.lr),
                      loss_fn, X_train, Y_train, args.batch_size, 1, args.seed, device, max_batches=3)
    del warm

    if args.train_mode == "online":
        settings_opt = {"optimizer": "online SGD", "optimizer_batch_size": 1, "feature_batch_size": 64}
        history, train_seconds = train_online_numpy(model, X_train, Y_train, args.epochs, args.lr,
                                                    args.seed, 64, device)
    else:
        opt = make_optimizer(args.optimizer, [p for p in model.parameters() if p.requires_grad], args.lr)
        settings_opt = {"optimizer": args.optimizer, "optimizer_batch_size": args.batch_size}
        if args.loader == "disk":
            history, train_seconds = train_from_disk(model, opt, loss_fn, data_dir / "train", args.batch_size,
                                                     args.epochs, args.seed, device, args.num_workers)
        else:
            history, train_seconds = train_batches(model, opt, loss_fn, X_train, Y_train, args.batch_size,
                                                   args.epochs, args.seed, device)

    metrics, blocks, scores, test_seconds = evaluate(model, X_test, Y_test, args.eval_block_size,
                                                     config.THRESHOLD, device)
    pipeline = perf_counter() - overall
    n_train, n_test = len(train_split.labels), len(test_split.labels)

    warnings = []
    if metrics["zero_output_fraction"] == 1.0:
        warnings.append("All outputs are zero: the output ReLU is inactive; results are not evidence of learning.")
    if min(np.sum(metrics["confusion_matrix"], axis=0)) == 0:
        warnings.append("Only one class was predicted on the test set.")

    label = args.label or "_".join([args.model, args.head, device.type, args.train_mode, args.loader])
    result = {
        "schema_version": 1,
        "part": "part3_torch",
        "label": label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "framework": "pytorch", "device": device.type,
        "dataset": data_dir.name, "num_train": n_train, "num_test": n_test,
        "epochs": args.epochs, "seed": args.seed,
        "accuracy": metrics["accuracy"],
        "train_seconds": train_seconds, "test_seconds": test_seconds,
        "test_images_per_second": n_test / test_seconds,
        "settings": {"model": args.model, "head": model.head, "train_mode": args.train_mode,
                     "loader": args.loader, "num_workers": args.num_workers,
                     "learning_rate": args.lr, **settings_opt,
                     "loss": "MSE" if model.head == "relu_mse" else "BCEWithLogits",
                     "image_size": config.IMG_SIZE, "classes": config.CLASSES,
                     "threshold": config.THRESHOLD, "eval_block_size": args.eval_block_size,
                     "dtype": "float32", "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                     "cudnn_benchmark": torch.backends.cudnn.benchmark,
                     "preprocessing": "ImageFolder (Pillow RGB), Resize BILINEAR, ToTensor (/255); NCHW"},
        "architecture": describe(model),
        "data": {"counts": {"train": train_split.counts, "test": test_split.counts},
                 "fingerprints_sha256": {"train": train_split.fingerprint, "test": test_split.fingerprint},
                 "fingerprint_format": "SHA256 of class/filename records: label<TAB>relative_path<TAB>file_sha256<LF> (same as Part II)",
                 "overlap_check": "No shared file names between train and test"},
        "timing": {"train_load_preprocess_seconds": train_load,
                   "test_load_preprocess_seconds": test_load,
                   "load_preprocess_seconds": train_load + test_load,
                   "host_to_device_seconds": h2d,
                   "train_seconds": train_seconds, "test_seconds": test_seconds,
                   "pipeline_seconds_before_export": pipeline,
                   "train_images_per_second": n_train * args.epochs / train_seconds,
                   "test_images_per_second": n_test / test_seconds,
                   "peak_device_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   "definition": ("perf_counter with device synchronization at both ends of every timer. "
                                  "train = sum of epoch timers (shuffle, forward, loss, backward, update"
                                  + (", and decoding images with ImageFolder + DataLoader" if args.loader == "disk" else "")
                                  + "); test = sum of block timers (forward, threshold, correct count). "
                                  "Image loading/decoding (preload mode), host-to-device copy, warm-up and export are timed separately or excluded. "
                                  "Warm-up: a few training steps on a throwaway copy of the model and one test block before timing.")},
        "metrics": metrics, "history": history, "test_blocks": blocks,
        "environment": environment(device),
        "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ["run.py", "model.py", "data.py"]},
        "warnings": warnings,
        "comparison_note": ("Part II trains one fixed random filter's FC layer with online SGD; CatDogCNN trains two "
                            "convolution layers and two FC layers with Adam. Use the numpy_equiv runs to compare "
                            "implementations on the same network."),
        "artifacts": {"model": weights.name, "predictions": predictions.name},
    }
    manifest = data_dir / "manifest.json"
    if manifest.exists():
        result["data"]["preparation_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    with predictions.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "true_label", "predicted_label", "score"])
        for name, y, s in zip(test_split.files, test_split.labels.tolist(), scores.tolist()):
            w.writerow([name, y, int(s > config.THRESHOLD), s])
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Test accuracy: {100 * metrics['accuracy']:.2f}% | train: {train_seconds:.3f}s | "
          f"test: {test_seconds:.4f}s ({n_test / test_seconds:,.0f} images/s)")
    print(f"Saved {output}")
    for w_ in warnings:
        print(f"WARNING: {w_}")
    return result


if __name__ == "__main__":
    main()
