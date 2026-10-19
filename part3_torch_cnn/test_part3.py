"""Checks that Part III computes the same thing as Part II where it should.

Run from the repository root (needs data_small/ from data_prep/prepare_data.py):
    python -m unittest part3_torch_cnn.test_part3 -v
"""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from part2_numpy_cnn import data as p2data  # noqa: E402
from part2_numpy_cnn.model import NumpyCNN  # noqa: E402
from part3_torch_cnn import data as p3data  # noqa: E402
from part3_torch_cnn.model import CatDogCNN, NumpyEquivalentCNN  # noqa: E402
from part3_torch_cnn.run import train_online_numpy  # noqa: E402

TEST_DIR = ROOT / config.DATA_DIR / "test"
HAVE_DATA = TEST_DIR.is_dir()


@unittest.skipUnless(HAVE_DATA, "run data_prep/prepare_data.py first")
class SameDataAsPart2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2 = p2data.load_split(TEST_DIR, config.CLASSES, config.IMG_SIZE)
        cls.p3 = p3data.load_split(TEST_DIR, config.CLASSES, config.IMG_SIZE)

    def test_same_files_labels_and_fingerprint(self):
        self.assertEqual(self.p3.files, self.p2.files)
        self.assertEqual(self.p3.labels.tolist(), self.p2.labels.tolist())
        self.assertEqual(self.p3.fingerprint, self.p2.fingerprint)

    def test_same_pixels(self):
        # Part II keeps (N, H, W, C); ToTensor gives (N, C, H, W).
        diff = np.abs(self.p3.images.permute(0, 2, 3, 1).numpy() - self.p2.images).max()
        self.assertLessEqual(diff, 1e-7)


class SameNetworkAsPart2(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.images = rng.random((16, config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.float32)
        self.np_model = NumpyCNN(config.IMG_SIZE, seed=542)
        self.t_model = NumpyEquivalentCNN(config.IMG_SIZE, seed=542)
        self.x = torch.from_numpy(self.images).permute(0, 3, 1, 2).contiguous()

    def test_same_initial_weights(self):
        w = self.t_model.conv.weight.detach().numpy()[0].transpose(1, 2, 0)
        np.testing.assert_array_equal(w, self.np_model.conv_filter)
        np.testing.assert_array_equal(self.t_model.fc.weight.detach().numpy().T, self.np_model.W_fc)

    def test_same_features(self):
        np.testing.assert_allclose(self.t_model.features(self.x).detach().numpy(),
                                   self.np_model.features(self.images), rtol=1e-5, atol=1e-6)

    def test_same_training(self):
        # Two epochs of Part II's online SGD on the same 24 images.
        labels = (np.arange(24) % 2).astype(np.int64)
        imgs = np.random.default_rng(1).random((24, config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.float32)
        ref = NumpyCNN(config.IMG_SIZE, seed=542)
        order_rng = np.random.default_rng(542)
        for _ in range(2):
            order = order_rng.permutation(24)
            for off in range(0, 24, 64):
                ids = order[off:off + 64]
                for flat, y in zip(ref.features(imgs[ids]), labels[ids]):
                    ref.train_one(flat, y, 0.001)
        model = NumpyEquivalentCNN(config.IMG_SIZE, seed=542)
        X = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()
        train_online_numpy(model, X, torch.from_numpy(labels).float(), 2, 0.001, 542, 64, torch.device("cpu"))
        np.testing.assert_allclose(model.fc.weight.detach().numpy().T, ref.W_fc, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(model.fc.bias.detach().numpy(), ref.b_fc, rtol=1e-4, atol=1e-6)


class CatDogShapes(unittest.TestCase):
    def test_output_shape_and_heads(self):
        x = torch.rand(4, 3, config.IMG_SIZE, config.IMG_SIZE)
        out = CatDogCNN(config.IMG_SIZE, head="relu_mse")(x)
        self.assertEqual(tuple(out.shape), (4,))
        self.assertTrue(bool((out >= 0).all()))
        logits = CatDogCNN(config.IMG_SIZE, head="bce")(x)
        self.assertEqual(tuple(logits.shape), (4,))

    def test_channels_double(self):
        m = CatDogCNN(config.IMG_SIZE)
        self.assertEqual((m.conv1.in_channels, m.conv1.out_channels), (3, 16))
        self.assertEqual((m.conv2.in_channels, m.conv2.out_channels), (16, 32))


if __name__ == "__main__":
    unittest.main()
