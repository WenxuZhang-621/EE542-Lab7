# Part II: NumPy CNN

Implementation of Lab 7 Part II (PDF pp. 10-12). Uses A's shared `config.py`
and prepared dataset. Two comparison files are included in `results/`:
`part2_numpy.json` (original baseline) and `part2_numpy_summary.json` (six-run
summary). The report, full predictions and model checkpoints are shared with
teammates separately.

## Run from the repository root

Prepare `data_small/` using the root README and A's `data_prep/prepare_data.py`.
Then run:

```bash
python -m pip install numpy pillow
python part2_numpy_cnn/run.py --output results/part2_my_run.json
python -m unittest part2_numpy_cnn.test_model -v
```

The default uses shared settings: 64x64 RGB, seed 542, 10 epochs, cat=0,
dog=1 and output > 0.5 for dog. Dataset paths are resolved from the repository
root. Inputs are converted to RGB, resized with bilinear interpolation and
scaled to float32 values in [0,1].

The runner generates a result JSON, a prediction CSV and a model NPZ. It never
overwrites results by default. The example uses a new output filename because
the original `results/part2_numpy.json` is already committed. Choose a different
filename for each independent run.

## Model and corrections

`64x64x3 -> valid 3x3x3 convolution -> ReLU -> 2x2 max pool -> flatten(961)
-> FC(1) -> ReLU`

One random convolution filter remains fixed. Only 961 FC weights and one bias
are trained with online SGD, MSE and learning rate 0.001. Convolution and
pooling use vectorized NumPy operations. Feature groups of 64 still receive
one SGD update per image; features are recomputed every epoch.

The handout's FC dimension is corrected to 961. A's stratified train/test
folders replace the class-ordered 80/20 slicing shown in the handout.
Filename and byte-hash checks reject train/test overlap.

## Repeated experiments

For a fresh six-run experiment, create a new seed-542 baseline and give the
repeat script its path. Both the baseline model and prediction CSV generated
beside that JSON are required; the two committed JSON files alone are not
enough to rerun the repeat protocol.

```bash
python part2_numpy_cnn/run.py --output results/part2_local.json
python part2_numpy_cnn/repeat_experiments.py --baseline results/part2_local.json --runs-dir results/part2_local_repeats --summary results/part2_local_summary.json
```

This preserves the baseline, adds two seed-542 repeats and three supplementary
seeds (543, 544, 545), and writes the requested summary JSON plus CSV.
The original data split stays fixed. Runs are sequential in fresh processes;
matching existing results are reused. Source/data/configuration mismatches
are rejected. Use new `--runs-dir` and `--summary` paths for another repeat set.

## Files

- `model.py`: fixed convolution, ReLU, pooling, FC and SGD.
- `data.py`: image loading, preprocessing and overlap checks.
- `run.py`: training, block evaluation, timing and result export.
- `repeat_experiments.py`: repeated measurements and seed sensitivity.
- `test_model.py`: ten numerical and integration checks.

## Comparison with Part III

Results record data fingerprints, settings, environment and timer boundaries.
`accuracy` is a fraction in [0,1]. Training/test compute times exclude data
loading/preprocessing and export. CUDA timing must synchronize the device.
Part III has two trainable convolution layers; Part II has one fixed filter.
The requested comparison therefore includes architecture/training differences,
not only GPU hardware effects. Full result interpretation is in the separate
teammate handoff package.
