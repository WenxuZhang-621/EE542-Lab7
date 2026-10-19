# EE542 Lab 7 — Artificial Neural Network Tutorial

NumPy MLP (Part I), NumPy CNN (Part II) and PyTorch GPU CNN (Part III) on the
Hugging Face `microsoft/cats_vs_dogs` dataset.

```text
EE542-Lab7/
├── README.md
├── requirements.txt
├── .gitignore
├── part1_mlp/                 # A: Part I
│   ├── mlp.py                 #   2-layer MLP with SGD and backprop
│   └── run_examples.py        #   1D regression, rings, three blobs
├── data_prep/                 # A: Part II data
│   └── prepare_data.py        #   parquet -> data/ and data_small/
├── part2_numpy_cnn/           # B: Part II NumPy CNN, evaluation, timing
├── part3_torch_cnn/           # C: Part III PyTorch CatDogCNN on GPU
├── compare/                   # C: NumPy vs GPU table from results/*.json
├── results/                   # everyone: run outputs and result JSON
│   ├── part1_mlp_output.txt
│   └── data_manifest.json
├── data/                      # generated, not committed
│   ├── train/{cat,dog}/
│   ├── test/{cat,dog}/
│   └── manifest.json
└── data_small/                # generated, not committed
    ├── train/{cat,dog}/
    └── test/{cat,dog}/
```

Edit only your own folder. Write a clear commit message that says what you changed.

## Setup

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Part III also needs PyTorch and torchvision with CUDA. The AWS Deep Learning AMI (PyTorch) already has them.

## Part I — MLP

```bash
cd part1_mlp
python run_examples.py
```

Pure Python and NumPy; nothing to compile. Saved output: `results/part1_mlp_output.txt`.

## Data preparation

1. Download both parquet files (about 721 MB) from
   <https://huggingface.co/datasets/microsoft/cats_vs_dogs/tree/main/data> into a folder, e.g. `../raw/`:

   ```bash
   mkdir -p ../raw
   for f in train-00000-of-00002 train-00001-of-00002; do
     curl -L -o ../raw/$f.parquet https://huggingface.co/datasets/microsoft/cats_vs_dogs/resolve/main/data/$f.parquet
   done
   ```

2. From the repo root:

   ```bash
   python data_prep/prepare_data.py --parquet-dir ../raw
   ```

The script is deterministic (seed 542), so every member gets exactly the same files. It takes about 16 s on an M2 Max.

| Folder | cat | dog | Use |
|---|---|---|---|
| `data/train` | 9393 | 9335 | full training split |
| `data/test` | 2348 | 2334 | full unseen test split |
| `data_small/train` | 1000 | 1000 | subset of `data/train` |
| `data_small/test` | 500 | 500 | subset of `data/test` |

All 23,410 images decode; none were skipped. Train and test never overlap.
`data/manifest.json` records the parquet SHA-256 hashes, seed and counts.
`data/` and `data_small/` are not committed.

## Interface conventions for Part II and Part III

These rules keep the NumPy and GPU numbers comparable.

1. **Images.** Read from `data_small/` for the head-to-head comparison, and optionally also from `data/`.
   Use labels cat=0 and dog=1, which is `ImageFolder` order. Convert to RGB, resize to 64×64 and scale to [0, 1].
2. **Model.** Follow the PDF architecture for each part, with MSE loss and a ReLU output. Predict dog when the output is > 0.5.
   Use these shared settings for the comparison run:

   | Setting | Part II (NumPy) | Part III (PyTorch) | Source |
   |---|---|---|---|
   | Dataset | `data_small/` (2000 train / 1000 test) | same | group convention |
   | Input | 64×64 RGB, scaled to [0, 1] | same | PDF p.11, p.13 |
   | Epochs | 10 | 10 | PDF p.12 for NumPy; matched for GPU |
   | Learning rate | 0.001 (SGD, FC layer only) | 0.001 (Adam) | PDF p.12, p.14 |
   | Batch size | 1 (per image, as in the PDF loop) | 64 | PDF p.12; GPU value is a group choice |
   | Random seed | 542 (`np.random.seed`) | 542 (`torch.manual_seed`) | group convention |
   | Test batch size | 1 | 64 | — |

   Record the actual values in the result JSON. If you change one, note it in the JSON and tell the group.
3. **Timing.** Use `time.perf_counter()`, and report these three times separately:
   - `load_time_s` — decode and resize all images into memory once;
   - `train_time_s` — all epochs, excluding image decoding;
   - `test_time_s` — one forward pass over the whole test split.

   On GPU, call `torch.cuda.synchronize()` before reading the clock. Run the comparison with the same number of epochs on both sides.
4. **Result file.** Write `results/part2_numpy.json` or `results/part3_torch.json`:

   ```json
   {
     "part": "part2_numpy",
     "dataset": "data_small",
     "device": "cpu (Apple M2 Max)",
     "n_train": 2000,
     "n_test": 1000,
     "epochs": 10,
     "batch_size": 1,
     "load_time_s": 0.0,
     "train_time_s": 0.0,
     "test_time_s": 0.0,
     "train_imgs_per_s": 0.0,
     "test_imgs_per_s": 0.0,
     "test_accuracy": 0.0,
     "started_at": "2026-10-19T10:00:00-07:00"
   }
   ```

   Compute `train_imgs_per_s` as `n_train * epochs / train_time_s`, and `test_imgs_per_s` as `n_test / test_time_s`.

## Known issues in the lab handout code

- **PDF p.11, wrong FC size.** A 3×3 valid convolution on 64×64 gives 62×62, and 2×2 pooling then gives 31×31 = 961 features.
  The handout sizes `W_fc` as `32*32*3`, which does not match. Use `31*31`, or pad the convolution to get 32×32.
- **PDF p.11, test set is all dogs.** `load_data` reads all cats first and does not shuffle, so the 80/20 split puts only dogs in the test set.
  Use the `data/test` or `data_small/test` folders instead.
