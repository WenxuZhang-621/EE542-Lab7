"""Organize the HF microsoft/cats_vs_dogs parquet files into image folders (Lab7 Part II, PDF p.10).

Output (shared by Part II and Part III so both use the same images):

    data/train/{cat,dog}/        full split, 80% of every class
    data/test/{cat,dog}/         full split, remaining 20% (unseen)
    data_small/train/{cat,dog}/  fixed subset of data/train (default 1000 per class)
    data_small/test/{cat,dog}/   fixed subset of data/test  (default 500 per class)
    data/manifest.json           counts, seed, source file hashes

The test folders keep cat/dog subfolders so accuracy can be computed and
torchvision.datasets.ImageFolder can read them. Label order: cat=0, dog=1.

Usage (from repo root):
    python data_prep/prepare_data.py --parquet-dir ../raw
Download the two parquet files first from
https://huggingface.co/datasets/microsoft/cats_vs_dogs/tree/main/data
"""
import argparse
import glob
import hashlib
import io
import json
import os
import shutil

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

CLASSES = ["cat", "dog"]  # HF ClassLabel names; index == label


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(parquet_files):
    """Return a list of (row_id, label, jpeg_bytes) for decodable images."""
    rows, skipped = [], []
    row_id = 0
    for path in parquet_files:
        table = pq.read_table(path, columns=["image", "labels"])
        images = table.column("image").to_pylist()
        labels = table.column("labels").to_pylist()
        for img, label in zip(images, labels):
            data = img["bytes"]
            try:
                with Image.open(io.BytesIO(data)) as im:
                    im.convert("RGB").load()
                rows.append((row_id, int(label), data))
            except Exception:
                skipped.append(row_id)
            row_id += 1
    return rows, skipped


def write_split(rows, out_dir):
    for cls in CLASSES:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)
    for row_id, label, data in rows:
        name = f"{CLASSES[label]}.{row_id:05d}.jpg"
        with open(os.path.join(out_dir, CLASSES[label], name), "wb") as f:
            f.write(data)


def count(split_rows):
    return {cls: sum(1 for _, l, _ in split_rows if l == i) for i, cls in enumerate(CLASSES)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet-dir", required=True, help="folder with train-*.parquet")
    ap.add_argument("--out", default="data")
    ap.add_argument("--small-out", default="data_small")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--small-train-per-class", type=int, default=1000)
    ap.add_argument("--small-test-per-class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=542)
    args = ap.parse_args()

    parquet_files = sorted(glob.glob(os.path.join(args.parquet_dir, "train-*.parquet")))
    if not parquet_files:
        raise SystemExit(f"no train-*.parquet in {args.parquet_dir}")
    for d in (args.out, args.small_out):
        if os.path.exists(d):
            raise SystemExit(f"{d}/ already exists; remove it first to regenerate")

    rows, skipped = load_rows(parquet_files)
    rng = np.random.default_rng(args.seed)

    # Stratified split: shuffle each class, then cut, so train and test never overlap
    # and both classes appear in every split (the PDF's load_data split would put
    # only dogs in the test set because it does not shuffle).
    split = {"train": [], "test": [], "small_train": [], "small_test": []}
    for label in range(len(CLASSES)):
        cls_rows = [r for r in rows if r[1] == label]
        order = rng.permutation(len(cls_rows))
        cls_rows = [cls_rows[i] for i in order]
        n_test = int(round(len(cls_rows) * args.test_frac))
        test, train = cls_rows[:n_test], cls_rows[n_test:]
        split["train"] += train
        split["test"] += test
        split["small_train"] += train[:args.small_train_per_class]
        split["small_test"] += test[:args.small_test_per_class]

    write_split(split["train"], os.path.join(args.out, "train"))
    write_split(split["test"], os.path.join(args.out, "test"))
    write_split(split["small_train"], os.path.join(args.small_out, "train"))
    write_split(split["small_test"], os.path.join(args.small_out, "test"))

    manifest = {
        "source": "https://huggingface.co/datasets/microsoft/cats_vs_dogs",
        "parquet_sha256": {os.path.basename(p): sha256(p) for p in parquet_files},
        "seed": args.seed,
        "labels": {cls: i for i, cls in enumerate(CLASSES)},
        "decodable_images": len(rows),
        "skipped_row_ids": skipped,
        "counts": {k: count(v) for k, v in split.items()},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    shutil.copy(os.path.join(args.out, "manifest.json"), os.path.join(args.small_out, "manifest.json"))
    print(json.dumps({k: manifest[k] for k in ("decodable_images", "counts")}, indent=2))
    print(f"skipped {len(skipped)} undecodable rows")


if __name__ == "__main__":
    main()
