# Part III: PyTorch CNN on the GPU

Implementation of Lab 7 Part III (PDF pp. 13-14). Uses A's `config.py` and
prepared data, and writes results in the same JSON format as Part II so
`compare/compare.py` can put them side by side.

## Run from the repository root

```bash
pip install -r requirements.txt
pip install torch torchvision          # CUDA build for the GPU machine, see below
python -m unittest part3_torch_cnn.test_part3 -v
python part3_torch_cnn/run.py                         # handout model on the GPU -> results/part3_torch.json
python part3_torch_cnn/run_experiments.py             # all experiments -> results/part3_runs/
python compare/compare.py                             # -> compare/README.md and compare/figures/
```

For an RTX 50-series GPU (Blackwell, sm_120) use a CUDA 12.8 or newer build,
e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.

`run.py` never overwrites results unless `--overwrite` is given. Besides the
JSON it writes `<name>_model.pt` and `<name>_predictions.csv` next to it.

## What the handout asks for, and where it is

| Step | Where |
|---|---|
| Imports: torch, nn, functional, DataLoader, datasets, transforms | `model.py`, `data.py`, `run.py` |
| `device = cuda if available else cpu` | `run.py`, `pick_device` (`--device auto`) |
| Resize to 64x64, ToTensor, `ImageFolder`, `DataLoader` | `data.py`; one batch's shape is printed at start-up |
| `CatDogCNN`: two conv layers 3 -> 16 -> 32 with ReLU and max pooling, two FC layers, ReLU everywhere | `model.py` |
| `model.to(device)`, tensors on the same device | `run.py` |
| `nn.MSELoss`, `Adam(lr=0.001)` | `run.py` defaults (`--head relu_mse --optimizer adam --lr 1e-3`) |
| Training loop: zero_grad, forward, loss, backward, step | `train_batches` |
| Evaluation under `torch.no_grad()`, accuracy, timing on blocks of unseen images | `evaluate` |
| NumPy vs GPU comparison | `compare/compare.py` -> `compare/README.md` |

## Model

```
CatDogCNN: 3x64x64 -> conv 3x3 (16) -> ReLU -> maxpool 2 -> conv 3x3 (32) -> ReLU -> maxpool 2
           -> flatten (8192) -> FC 128 -> ReLU -> FC 1 -> ReLU (handout)  or  logit (--head bce)
```

Padding 1 keeps each convolution's output the size of its input, so the two
pools give 16x16x32. 1,053,921 trainable parameters (conv 448 + 4,640, FC 1,048,704 + 129).

## Timing

Every timer synchronizes the GPU at both ends: kernels run asynchronously,
so without it a timer only measures how long it took to queue the work.
Training and test timers cover computation only, as in Part II; decoding the
images and the host-to-device copy are recorded separately in `timing`.
Before timing, a few training steps run on a throwaway copy of the model and
one test block runs once, so one-off costs (loading kernels, choosing cuDNN
algorithms) are not counted. By default the decoded images are kept in GPU
memory; `--loader disk` instead reads them with ImageFolder + DataLoader every
epoch, to measure the input pipeline.

## Experiments (`run_experiments.py`)

| Group | Question |
|---|---|
| `main` | The handout's model on the GPU (`results/part3_torch.json`) |
| `heads` | ReLU + MSE (handout) vs sigmoid + BCE, four seeds each |
| `cpu_gpu` | The same CatDogCNN on the CPU |
| `same_net` | Part II's network in NumPy (Part II's own code), PyTorch CPU and PyTorch GPU, with Part II's online SGD and with mini-batches |
| `pipeline` | Images read from disk every epoch (0 and 4 DataLoader workers) vs kept in GPU memory |
| `batch` | Throughput against batch size |
| `full` | The full `data/` split on GPU and CPU |

`NumpyEquivalentCNN` is Part II's network written in PyTorch with the same
initial weights. With `--train-mode online` it repeats Part II's training
step for step; `test_part3.py` checks that it produces the same weights as
`part2_numpy_cnn`, and that both parts read the same files and pixels.

## Finding: the handout's ReLU output can stop learning

With ReLU on the single output and MSE loss, if the output becomes zero for
every image the gradient is zero and training stops; the network then
predicts one class. In development runs on `data_small/` this happened for
three of four seeds (542, 543, 545), while seed 544 reached 68.9%. With a
logit output and `BCEWithLogitsLoss` (`--head bce`) all four seeds learned
(67.1-69.6%). Both versions are reported; the handout's version is the main
run.

## Files

- `model.py`: `CatDogCNN` and `NumpyEquivalentCNN`.
- `data.py`: ImageFolder loading, Part II-compatible data fingerprint.
- `run.py`: training, block evaluation, timing and result export.
- `run_experiments.py`: the experiment list, one fresh process per run.
- `test_part3.py`: checks against Part II and model shape checks.
