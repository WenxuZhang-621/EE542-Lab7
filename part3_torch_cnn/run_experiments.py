"""Run every Part III experiment on one machine, one fresh process each.

    python part3_torch_cnn/run_experiments.py              # all experiments
    python part3_torch_cnn/run_experiments.py --only main  # one group

Results go to results/part3_runs/<name>.json (existing results are kept and
skipped); the handout's main run is also written to results/part3_torch.json.
Then build the comparison with:  python compare/compare.py
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

RUNS = "results/part3_runs"
SEEDS = [config.SEED, config.SEED + 1, config.SEED + 2, config.SEED + 3]


def experiments():
    """(group, name, script, args). Groups:
    main      the handout's CatDogCNN (ReLU output, MSE, Adam 1e-3) on the GPU
    heads     handout head vs sigmoid/BCE head, four seeds each, GPU
    cpu_gpu   the same CatDogCNN on the CPU, for the device comparison
    same_net  Part II's network and algorithm: NumPy (Part II code) vs PyTorch CPU vs GPU
    pipeline  reading images from disk every epoch vs preloaded tensors
    batch     training throughput against batch size
    full      the full data/ split instead of data_small/
    """
    p3 = "part3_torch_cnn/run.py"
    p2 = "part2_numpy_cnn/run.py"
    out = []
    out.append(("main", "main", p3, ["--output", "results/part3_torch.json", "--label", "catdog_handout_gpu"]))
    for head in ("relu_mse", "bce"):
        for s in SEEDS:
            out.append(("heads", f"catdog_{head}_gpu_seed{s}", p3,
                        ["--device", "cuda", "--head", head, "--seed", str(s)]))
    for head in ("relu_mse", "bce"):
        out.append(("cpu_gpu", f"catdog_{head}_cpu_seed{config.SEED}", p3,
                    ["--device", "cpu", "--head", head]))
    out.append(("same_net", "numpy_part2_code", p2, []))
    for dev, tag in (("cpu", "cpu"), ("cuda", "gpu")):
        out.append(("same_net", f"numpy_equiv_online_{tag}", p3,
                    ["--device", dev, "--model", "numpy_equiv", "--train-mode", "online", "--optimizer", "sgd"]))
        out.append(("same_net", f"numpy_equiv_batch64_{tag}", p3,
                    ["--device", dev, "--model", "numpy_equiv", "--optimizer", "sgd", "--batch-size", "64"]))
    for workers in (0, 4):
        out.append(("pipeline", f"catdog_bce_gpu_disk_w{workers}", p3,
                    ["--device", "cuda", "--head", "bce", "--loader", "disk", "--num-workers", str(workers)]))
    for bs in (16, 64, 256, 1024):
        out.append(("batch", f"catdog_bce_gpu_bs{bs}", p3,
                    ["--device", "cuda", "--head", "bce", "--batch-size", str(bs)]))
    for dev, tag in (("cuda", "gpu"), ("cpu", "cpu")):
        out.append(("full", f"catdog_bce_{tag}_full", p3,
                    ["--device", dev, "--head", "bce", "--data-dir", config.FULL_DATA_DIR]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", help="run only these groups")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args()
    (ROOT / RUNS).mkdir(parents=True, exist_ok=True)
    for group, name, script, extra in experiments():
        if args.only and group not in args.only:
            continue
        if "--output" in extra:
            output = extra[extra.index("--output") + 1]
            cmd = [sys.executable, script, *extra]
        else:
            output = f"{RUNS}/{name}.json"
            cmd = [sys.executable, script, *extra, "--output", output]
            if script.endswith("part3_torch_cnn/run.py"):
                cmd += ["--label", name]
        if (ROOT / output).exists():
            print(f"[skip] {name}: {output} exists")
            continue
        print(f"[run]  {name}: {' '.join(cmd[1:])}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
