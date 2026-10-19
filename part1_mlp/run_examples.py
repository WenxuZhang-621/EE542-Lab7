"""Run the three Part I application examples (PDF p.9-10).

Usage (from repo root):  python part1_mlp/run_examples.py
"""
import numpy as np

from mlp import MLP2, train


def example1_regression():
    print("=== Example 1: 1D regression (piecewise-linear) ===")
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (400, 1))

    def target(x):
        return np.maximum(0, 0.5 * x + 0.2) + 0.3 * np.maximum(0, -x + 0.5)

    y = target(X) + 0.05 * rng.normal(size=(400, 1))

    # Model: 1->32->1
    m_reg = MLP2(in_dim=1, hidden_dim=32, out_dim=1, mode='regression', seed=1)
    train(m_reg, X, y, epochs=500, lr=1e-2, batch_size=64, weight_decay=1e-4)

    xt = np.linspace(-2, 2, 9).reshape(-1, 1)
    pred = m_reg.forward(xt)
    print("      x   target     pred")
    print(np.array2string(np.hstack([xt, target(xt), pred]), precision=4,
                          suppress_small=True))
    print(f"max |pred - target| on test points: {np.abs(pred - target(xt)).max():.4f}")


def rings(n=600, inner_r=0.5, gap=0.2, noise=0.06, seed=0):
    rng = np.random.default_rng(seed)
    n2 = n // 2
    theta1 = rng.uniform(0, 2 * np.pi, n2)
    r1 = inner_r + noise * rng.normal(size=n2)
    x1 = np.c_[r1 * np.cos(theta1), r1 * np.sin(theta1)]
    theta2 = rng.uniform(0, 2 * np.pi, n - n2)
    r2 = inner_r + gap + noise * rng.normal(size=n - n2)
    x2 = np.c_[r2 * np.cos(theta2), r2 * np.sin(theta2)]
    X = np.vstack([x1, x2])
    y = np.array([0] * n2 + [1] * (n - n2))
    return X, y


def example2_binary():
    print("\n=== Example 2: binary classification (concentric rings) ===")
    Xb, yb = rings(n=800, inner_r=0.6, gap=0.5, noise=0.07, seed=1)
    m_bin = MLP2(in_dim=2, hidden_dim=64, out_dim=2, mode='classification', seed=2)
    train(m_bin, Xb, yb, epochs=200, lr=5e-3, batch_size=64, weight_decay=1e-4)
    preds = m_bin.forward(Xb).argmax(axis=1)
    print("train accuracy:", (preds == yb).mean())


def three_blobs(n=900, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([[0, 0], [2.5, 0.5], [-2.0, 1.5]])
    cov = np.array([[0.4, 0.0], [0.0, 0.4]])
    Xs, ys = [], []
    for k, m in enumerate(means):
        Xs.append(rng.multivariate_normal(m, cov, size=n // 3))
        ys.append(np.full(n // 3, k))
    return np.vstack(Xs), np.hstack(ys)


def example3_multiclass():
    print("\n=== Example 3: multiclass classification (3 blobs) ===")
    Xm, ym = three_blobs(900, seed=4)
    m_multi = MLP2(in_dim=2, hidden_dim=64, out_dim=3, mode='classification', seed=3)
    train(m_multi, Xm, ym, epochs=200, lr=5e-3, batch_size=64)
    print("train acc:", (m_multi.forward(Xm).argmax(1) == ym).mean())


if __name__ == "__main__":
    example1_regression()
    example2_binary()
    example3_multiclass()
