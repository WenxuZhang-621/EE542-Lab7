"""NumPy implementation of the Part II architecture, without autograd."""
import numpy as np


def conv2d(images, kernel):
    """Valid stride-1 cross-correlation, matching the handout's conv2d.

    Input: (N,H,W,C). One (3,3,C) filter produces (N,H-2,W-2).
    Vectorizing the spatial loops changes execution speed, not the operation.
    """
    if images.ndim != 4 or kernel.shape != (3, 3, images.shape[-1]):
        raise ValueError("Expected NHWC images and one 3x3xC filter")
    if min(images.shape[1:3]) < 3:
        raise ValueError("Images must be at least 3x3")
    windows = np.lib.stride_tricks.sliding_window_view(images, (3, 3), axis=(1, 2))
    # windows: N, H-2, W-2, C, kernel-H, kernel-W (a view, not an im2col copy).
    return np.einsum("nhwcij,ijc->nhw", windows, kernel, optimize=False)


def maxpool(images):
    """Non-overlapping 2x2 max pooling; discard incomplete edge windows."""
    n, h, w = images.shape
    ph, pw = h // 2, w // 2
    if min(ph, pw) < 1:
        raise ValueError("Feature maps must be at least 2x2")
    return images[:, :2*ph, :2*pw].reshape(n, ph, 2, pw, 2).max(axis=(2, 4))


class NumpyCNN:
    """RGB -> one fixed convolution -> ReLU -> pool -> trainable FC -> ReLU."""

    def __init__(self, image_size=64, seed=542):
        if image_size < 4:
            raise ValueError("image_size must be at least 4")
        self.image_size = image_size
        rng = np.random.default_rng(seed)
        self.conv_filter = rng.normal(0.0, 0.1, (3, 3, 3)).astype(np.float32)
        pooled_size = (image_size - 3 + 1) // 2
        self.feature_dim = pooled_size * pooled_size
        self.W_fc = rng.normal(0.0, 0.01, (self.feature_dim, 1)).astype(np.float32)
        self.b_fc = np.zeros(1, dtype=np.float32)

    def features(self, images):
        if images.shape[1:] != (self.image_size, self.image_size, 3):
            raise ValueError(f"Expected Nx{self.image_size}x{self.image_size}x3 images")
        return maxpool(np.maximum(conv2d(images, self.conv_filter), 0.0)).reshape(
            len(images), self.feature_dim
        )

    def predict_features(self, features):
        return np.maximum(features @ self.W_fc + self.b_fc, 0.0).ravel()

    def predict(self, images):
        return self.predict_features(self.features(images))

    def train_one(self, flat, label, learning_rate):
        """One online SGD update, with the handout's MSE and output ReLU."""
        z = float((flat @ self.W_fc + self.b_fc).item())
        prediction = max(z, 0.0)
        error = prediction - float(label)
        grad = 2.0 * error if z > 0.0 else 0.0
        self.W_fc[:, 0] -= learning_rate * grad * flat
        self.b_fc[0] -= learning_rate * grad
        return error * error

    def save(self, path):
        np.savez_compressed(path, conv_filter=self.conv_filter, W_fc=self.W_fc,
                            b_fc=self.b_fc, image_size=self.image_size)
