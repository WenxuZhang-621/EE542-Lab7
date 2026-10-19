"""Settings shared by every part of Lab 7. Import with `import config`."""

DATA_DIR = "data_small"   # dataset for the NumPy vs GPU comparison
FULL_DATA_DIR = "data"    # full 80/20 split
CLASSES = ["cat", "dog"]  # label 0 = cat, 1 = dog (same order as ImageFolder)
IMG_SIZE = 64             # images resized to 64x64 RGB, scaled to [0, 1]
SEED = 542
EPOCHS = 10               # same for the NumPy and GPU runs
THRESHOLD = 0.5           # output > 0.5 -> dog
