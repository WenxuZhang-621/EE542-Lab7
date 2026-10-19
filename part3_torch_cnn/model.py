"""Part III models.

CatDogCNN is the network the handout asks for. NumpyEquivalentCNN is Part II's
NumPy network written in PyTorch, with the same initial weights, so the two
implementations can be compared on identical arithmetic.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CatDogCNN(nn.Module):
    """Two conv layers (3 -> 16 -> 32 channels), each with ReLU and 2x2 max
    pooling, then two fully connected layers.

    head="relu_mse": the handout's version, ReLU on the single output and MSE
    loss; output > threshold means dog.
    head="bce": no output activation; the output is a logit for
    BCEWithLogitsLoss, the usual choice for binary classification.
    """

    def __init__(self, image_size=64, hidden=128, head="relu_mse"):
        super().__init__()
        if head not in ("relu_mse", "bce"):
            raise ValueError("head must be relu_mse or bce")
        if image_size % 4:
            raise ValueError("image_size must be a multiple of 4 (two 2x2 pools)")
        self.head = head
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # 64x64 -> 64x64
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 32x32 -> 32x32
        self.flat_dim = 32 * (image_size // 4) * (image_size // 4)
        self.fc1 = nn.Linear(self.flat_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(torch.flatten(x, 1)))
        x = self.fc2(x).squeeze(1)
        return F.relu(x) if self.head == "relu_mse" else x

    def score(self, out):
        """Map the network output to a dog score compared with the threshold."""
        return out if self.head == "relu_mse" else torch.sigmoid(out)


class NumpyEquivalentCNN(nn.Module):
    """Part II's network: one fixed random 3x3x3 filter (valid, no bias) ->
    ReLU -> 2x2 max pool -> flatten -> FC(1) -> ReLU. Only the FC layer is
    trained.

    Weights are drawn exactly as part2_numpy_cnn.model.NumpyCNN draws them
    (same generator, same order, float32), so for the same seed both start
    from identical parameters.
    """

    head = "relu_mse"

    def __init__(self, image_size=64, seed=542):
        super().__init__()
        rng = np.random.default_rng(seed)
        conv = rng.normal(0.0, 0.1, (3, 3, 3)).astype(np.float32)        # (kh, kw, c)
        pooled = (image_size - 3 + 1) // 2
        self.feature_dim = pooled * pooled
        w_fc = rng.normal(0.0, 0.01, (self.feature_dim, 1)).astype(np.float32)
        self.conv = nn.Conv2d(3, 1, kernel_size=3, bias=False)
        self.fc = nn.Linear(self.feature_dim, 1)
        with torch.no_grad():
            # NumPy's (kh, kw, c) cross-correlation kernel -> PyTorch (out, c, kh, kw).
            self.conv.weight.copy_(torch.from_numpy(conv.transpose(2, 0, 1)[None]))
            self.fc.weight.copy_(torch.from_numpy(w_fc.T))
            self.fc.bias.zero_()
        self.conv.weight.requires_grad_(False)   # fixed filter, as in Part II

    def features(self, x):
        return torch.flatten(F.max_pool2d(F.relu(self.conv(x)), 2), 1)

    def forward_features(self, feats):
        return F.relu(self.fc(feats)).squeeze(1)

    def forward(self, x):
        return self.forward_features(self.features(x))

    def score(self, out):
        return out


def build(name, image_size, seed, head="relu_mse"):
    if name == "catdog":
        torch.manual_seed(seed)
        return CatDogCNN(image_size=image_size, head=head)
    if name == "numpy_equiv":
        if head != "relu_mse":
            raise ValueError("numpy_equiv only has the handout's ReLU/MSE head")
        return NumpyEquivalentCNN(image_size=image_size, seed=seed)
    raise ValueError(f"unknown model {name}")


def describe(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"class": type(model).__name__, "head": model.head,
            "trainable_parameters": trainable, "fixed_parameters": fixed,
            "layers": str(model)}
