"""Load A's existing split; never create a second split in Part II."""
from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Split:
    images: np.ndarray
    labels: np.ndarray
    files: list
    hashes: list
    fingerprint: str
    counts: dict


def load_split(root, classes, image_size):
    root = Path(root)
    paths, labels, counts = [], [], {}
    for label, cls in enumerate(classes):
        folder = root / cls
        if not folder.is_dir():
            raise ValueError(f"Missing {folder}; run A's data_prep/prepare_data.py first")
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError(f"No images in {folder}")
        paths.extend(files)
        labels.extend([label] * len(files))
        counts[cls] = len(files)
    images = np.empty((len(paths), image_size, image_size, 3), dtype=np.float32)
    names, hashes = [], []
    fingerprint = hashlib.sha256()
    for i, (path, label) in enumerate(zip(paths, labels)):
        name = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            with Image.open(path) as im:
                resized = im.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
                images[i] = np.asarray(resized, dtype=np.float32) / 255.0
        except Exception as exc:
            raise ValueError(f"Cannot decode {path}; no samples have been silently skipped") from exc
        names.append(name)
        hashes.append(digest)
        fingerprint.update(f"{label}\t{name}\t{digest}\n".encode("utf-8"))
    return Split(images, np.asarray(labels, dtype=np.int64), names, hashes,
                 fingerprint.hexdigest(), counts)


def assert_disjoint(train, test):
    if set(train.files) & set(test.files):
        raise ValueError("Train/test overlap: repeated source filename")
    if set(train.hashes) & set(test.hashes):
        raise ValueError("Train/test overlap: identical image file content")
