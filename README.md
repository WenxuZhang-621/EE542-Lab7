# EE542 Lab 7 — Artificial Neural Network Tutorial

NumPy MLP (Part I), NumPy CNN (Part II) and PyTorch GPU CNN (Part III) on Hugging Face `microsoft/cats_vs_dogs`.

```text
EE542-Lab7/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                  # shared settings: dataset, image size, labels, seed, epochs
├── part1_mlp/                 # A: Part I
│   ├── mlp.py                 #   2-layer MLP with SGD and backprop
│   └── run_examples.py        #   1D regression, rings, three blobs
├── data_prep/                 # A: Part II data
│   └── prepare_data.py        #   parquet -> data/ and data_small/
├── part2_numpy_cnn/           # B: Part II NumPy CNN, evaluation, timing
├── part3_torch_cnn/           # C: Part III PyTorch CatDogCNN on GPU
├── compare/                   # C: NumPy vs GPU table from results/*.json
├── results/                   # run outputs and result JSON
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

Edit only your own folder, and write a clear commit message.

## Run

```bash
pip install -r requirements.txt            # Python 3.10+; Part III also needs torch + torchvision (CUDA)

# Part I
cd part1_mlp && python run_examples.py && cd ..

# Data: download both parquet files (~721 MB) into ../raw, then split
mkdir -p ../raw
for f in train-00000-of-00002 train-00001-of-00002; do
  curl -L -o ../raw/$f.parquet https://huggingface.co/datasets/microsoft/cats_vs_dogs/resolve/main/data/$f.parquet
done
python data_prep/prepare_data.py --parquet-dir ../raw
```

The split is deterministic (seed 542), so everyone gets the same files.

- `data/`: full set, train 18,728 and test 4,682 images.
- `data_small/`: train 2,000 and test 1,000 images, balanced cat/dog.

Labels are cat=0 and dog=1. Train and test never overlap.

Shared settings live in `config.py`. Import them with `import config` instead of hard-coding the values.
