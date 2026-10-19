"""Numerical and integration checks. No real dataset or PyTorch required."""
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image

from part2_numpy_cnn.data import assert_disjoint, load_split
from part2_numpy_cnn.model import NumpyCNN, conv2d, maxpool
from part2_numpy_cnn.run import evaluate


class ModelTests(unittest.TestCase):
    def test_convolution_matches_independent_pixel_loops(self):
        rng = np.random.default_rng(42)
        images = rng.normal(size=(2, 7, 6, 3)).astype(np.float32)
        kernel = rng.normal(size=(3, 3, 3)).astype(np.float32)
        expected = np.empty((2, 5, 4), dtype=np.float32)
        for n in range(2):
            for i in range(5):
                for j in range(4):
                    expected[n, i, j] = np.sum(images[n, i:i+3, j:j+3] * kernel)
        np.testing.assert_allclose(conv2d(images, kernel), expected, atol=3e-6, rtol=2e-6)

    def test_pooling_discards_incomplete_edges(self):
        images = np.arange(2*5*7).reshape(2, 5, 7)
        expected = np.array([[[8, 10, 12], [22, 24, 26]],
                             [[43, 45, 47], [57, 59, 61]]])
        np.testing.assert_array_equal(maxpool(images), expected)

    def test_pdf_dimension_fix_and_full_forward(self):
        model = NumpyCNN(64, 542)
        self.assertEqual(model.feature_dim, 961)
        self.assertEqual(model.W_fc.shape, (961, 1))
        self.assertEqual(model.predict(np.zeros((2, 64, 64, 3), np.float32)).shape, (2,))

    def test_sgd_gradient_matches_finite_differences(self):
        model = NumpyCNN(4)
        model.W_fc = np.array([[0.2]], dtype=np.float64)
        model.b_fc = np.array([0.3], dtype=np.float64)
        flat = np.array([0.7], dtype=np.float64)
        label, epsilon, lr = 1.0, 1e-6, 0.01
        def loss(w, b):
            return (max(float(flat[0] * w + b), 0) - label)**2
        w, b = model.W_fc.item(), model.b_fc.item()
        numerical_w = (loss(w+epsilon, b)-loss(w-epsilon, b))/(2*epsilon)
        numerical_b = (loss(w, b+epsilon)-loss(w, b-epsilon))/(2*epsilon)
        original_filter = model.conv_filter.copy()
        model.train_one(flat, label, lr)
        self.assertAlmostEqual((w-model.W_fc.item())/lr, numerical_w, places=8)
        self.assertAlmostEqual((b-model.b_fc.item())/lr, numerical_b, places=8)
        np.testing.assert_array_equal(model.conv_filter, original_filter)

    def test_inactive_output_relu_has_zero_gradient(self):
        model = NumpyCNN(4)
        model.W_fc[:] = -1
        model.b_fc[:] = -1
        self.assertEqual(model.train_one(np.ones(1, np.float32), 1, 0.1), 1.0)
        np.testing.assert_array_equal(model.W_fc, [[-1]])
        np.testing.assert_array_equal(model.b_fc, [-1])

    def test_small_regression_example_learns(self):
        model = NumpyCNN(4)
        model.W_fc[:] = 0.1
        model.b_fc[:] = 0.1
        features = np.array([[0.0], [1.0]], dtype=np.float32)
        labels = np.array([0, 1])
        initial = np.mean((model.predict_features(features)-labels)**2)
        for _ in range(100):
            for flat, label in zip(features, labels):
                model.train_one(flat, label, 0.1)
        final = np.mean((model.predict_features(features)-labels)**2)
        self.assertLess(final, initial / 100)

    def test_checkpoint_preserves_predictions(self):
        model = NumpyCNN(6)
        images = np.random.default_rng(1).random((3, 6, 6, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.npz"
            model.save(path)
            with np.load(path) as saved:
                restored = NumpyCNN(int(saved["image_size"]), seed=999)
                for field in ["conv_filter", "W_fc", "b_fc"]:
                    setattr(restored, field, saved[field])
            np.testing.assert_array_equal(model.predict(images), restored.predict(images))


class DataAndEvaluationTests(unittest.TestCase):
    def test_loader_handles_grayscale_rgb_and_deterministic_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for cls, value in [("cat", 0), ("dog", 255)]:
                (root/cls).mkdir()
                Image.new("L", (8, 7), color=value).save(root/cls/"b.png")
                Image.new("RGB", (7, 8), color=(value,)*3).save(root/cls/"a.png")
            data = load_split(root, ["cat", "dog"], 64)
            self.assertEqual(data.images.shape, (4, 64, 64, 3))
            self.assertEqual(data.images.dtype, np.float32)
            self.assertEqual(data.files, ["cat/a.png", "cat/b.png", "dog/a.png", "dog/b.png"])
            np.testing.assert_array_equal(data.labels, [0, 0, 1, 1])
            self.assertEqual(data.images[2].mean(), 1.0)
            self.assertEqual(data.fingerprint, load_split(root, ["cat", "dog"], 64).fingerprint)

    def test_leakage_rejected_even_when_duplicate_is_renamed(self):
        train = SimpleNamespace(files=["cat/a.jpg"], hashes=["same-bytes"])
        test = SimpleNamespace(files=["cat/renamed.jpg"], hashes=["same-bytes"])
        with self.assertRaisesRegex(ValueError, "identical image"):
            assert_disjoint(train, test)
        test = SimpleNamespace(files=["cat/a.jpg"], hashes=["different-bytes"])
        with self.assertRaisesRegex(ValueError, "filename"):
            assert_disjoint(train, test)

    def test_uneven_test_blocks_use_sample_weighted_accuracy_and_threshold(self):
        model = SimpleNamespace(predict=lambda images: images.ravel())
        split = SimpleNamespace(images=np.array([[0], [.5], [.8], [.2], [.1]], np.float32),
                                labels=np.array([0, 0, 1, 0, 1]))
        with redirect_stdout(io.StringIO()):
            metrics, blocks, scores, seconds = evaluate(model, split, 2, 0.5)
        self.assertEqual([b["num_images"] for b in blocks], [2, 2, 1])
        self.assertEqual(metrics["accuracy"], 0.8)
        self.assertEqual(metrics["confusion_matrix"], [[3, 0], [1, 1]])
        self.assertEqual(metrics["correct"], sum(b["correct"] for b in blocks))


if __name__ == "__main__":
    unittest.main()
