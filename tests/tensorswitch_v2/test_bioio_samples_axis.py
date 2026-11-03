"""BIOIOReader names axes from BioIO's own dimension order, including the trailing S (samples per pixel)."""
import numpy as np
import pytest
import tifffile

pytest.importorskip("bioio")
pytest.importorskip("bioio_tifffile")
from tensorswitch_v2.readers.bioio_adapter import BIOIOReader  # noqa: E402


def write(path, arr, **kw):
    kw.setdefault("shaped", False)
    tifffile.imwrite(str(path), arr, **kw)
    return str(path)


def labels_and_data(reader):
    store = reader.get_tensorstore()
    return list(store.domain.labels), store.read().result()


def test_rgb_stack_keeps_samples_as_their_own_axis(tmp_path):
    arr = np.random.default_rng(0).integers(1, 250, (8, 32, 48, 3), dtype=np.uint8)
    labels, data = labels_and_data(BIOIOReader(write(tmp_path / "rgb.tif", arr, photometric="rgb")))
    assert labels[-3:] == ["y", "x", "s"] and data.shape[-3:] == (32, 48, 3)
    assert data.size == arr.size and np.array_equal(data.reshape(arr.shape), arr)


def test_plain_volume_is_unchanged(tmp_path):
    arr = np.random.default_rng(1).integers(1, 250, (6, 32, 48), dtype=np.uint8)
    labels, data = labels_and_data(BIOIOReader(write(tmp_path / "vol.tif", arr, photometric="minisblack", metadata={"axes": "ZYX"}, shaped=True)))
    assert labels == ["z", "y", "x"] and np.array_equal(data, arr)


def test_2d_rgb_image(tmp_path):
    arr = np.random.default_rng(2).integers(1, 250, (32, 48, 3), dtype=np.uint8)
    labels, data = labels_and_data(BIOIOReader(write(tmp_path / "img.tif", arr, photometric="rgb")))
    assert labels[-3:] == ["y", "x", "s"] and np.array_equal(data.reshape(arr.shape), arr)


def test_channel_stack_names_channels_not_samples(tmp_path):
    arr = np.random.default_rng(3).integers(1, 250, (2, 5, 32, 48), dtype=np.uint16)
    path = tmp_path / "czyx.ome.tif"
    tifffile.imwrite(str(path), arr, ome=True, metadata={"axes": "CZYX"})
    labels, data = labels_and_data(BIOIOReader(str(path)))
    assert labels == ["c", "z", "y", "x"] and np.array_equal(data, arr)


def test_selected_channel_is_dropped_from_the_axes(tmp_path):
    arr = np.random.default_rng(4).integers(1, 250, (2, 5, 32, 48), dtype=np.uint16)
    path = tmp_path / "sel.ome.tif"
    tifffile.imwrite(str(path), arr, ome=True, metadata={"axes": "CZYX"})
    labels, data = labels_and_data(BIOIOReader(str(path), channel_index=1))
    assert labels == ["z", "y", "x"] and np.array_equal(data, arr[1])
