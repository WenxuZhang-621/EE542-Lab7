"""Load A's split with torchvision.datasets.ImageFolder.

Preprocessing matches Part II: Pillow RGB, bilinear resize to IMG_SIZE x
IMG_SIZE, scaled to [0, 1]. ToTensor gives channel-first (C, H, W) float32,
where Part II keeps (H, W, C); the values are the same.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


def make_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])


def image_folder(split_dir, classes, image_size):
    ds = datasets.ImageFolder(split_dir, transform=make_transform(image_size))
    if ds.classes != list(classes):
        raise ValueError(f"{split_dir}: classes {ds.classes}, expected {list(classes)}")
    return ds


def fingerprint(ds):
    """SHA-256 over 'label<TAB>class/file<TAB>file_sha256<LF>' in ImageFolder
    order (class, then file name): the same records as Part II, so equal
    fingerprints mean both parts used the same files."""
    root = Path(ds.root)
    h = hashlib.sha256()
    for path, label in ds.samples:
        p = Path(path)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(f"{label}\t{p.relative_to(root).as_posix()}\t{digest}\n".encode("utf-8"))
    return h.hexdigest()


@dataclass
class Split:
    images: torch.Tensor      # (N, 3, H, W) float32, CPU
    labels: torch.Tensor      # (N,) int64, CPU
    files: list
    counts: dict
    fingerprint: str


def load_split(split_dir, classes, image_size, batch_size=256, num_workers=0):
    """Decode every image once with ImageFolder + DataLoader and stack them."""
    ds = image_folder(split_dir, classes, image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    images, labels = [], []
    for x, y in loader:
        images.append(x)
        labels.append(y)
    root = Path(ds.root)
    files = [Path(p).relative_to(root).as_posix() for p, _ in ds.samples]
    counts = {c: sum(1 for _, l in ds.samples if l == i) for i, c in enumerate(ds.classes)}
    return Split(torch.cat(images), torch.cat(labels), files, counts, fingerprint(ds))


def assert_disjoint(train, test):
    if set(f.split("/", 1)[1] for f in train.files) & set(f.split("/", 1)[1] for f in test.files):
        raise ValueError("train/test overlap: repeated file name")
