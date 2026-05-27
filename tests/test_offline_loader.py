import os
import numpy as np
from PIL import Image
from data.offline_loader import OfflineDatasetLoader


def make_rgb_png(path, size=(64, 48), color=(10, 20, 30)):
    im = Image.new('RGB', (size[0], size[1]), color)
    im.save(path)


def test_load_valid_images(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    # create valid png
    p1 = d / "img1.png"
    make_rgb_png(str(p1), size=(128, 96))

    # create valid npy HxWx3
    arr = np.zeros((96, 128, 3), dtype=np.uint8)
    p2 = d / "img2.npy"
    np.save(str(p2), arr)

    loader = OfflineDatasetLoader(str(d), cfg={'prevalidate': True})
    got = list(loader.load())
    assert len(got) == 2
    assert got[0].image.shape[2] == 3
    assert got[1].image.dtype == np.uint8


def test_invalid_npy_raises(tmp_path):
    d = tmp_path / "ds2"
    d.mkdir()
    # create invalid npy with shape (1,)
    arr = np.array([1])
    p1 = d / "bad.npy"
    np.save(str(p1), arr)

    try:
        OfflineDatasetLoader(str(d), cfg={'prevalidate': True})
        raised = False
    except RuntimeError:
        raised = True

    assert raised
