"""CZI reader on small generated files (the real CZI test file's symlink target is gone)."""
import numpy as np
import pytest

pytest.importorskip("pylibCZIrw")
from pylibCZIrw.czi import create_czi  # noqa: E402

from tensorswitch_v2.api.readers import Readers  # noqa: E402
from tensorswitch_v2.readers import CZIReader  # noqa: E402

VOLUME = (np.arange(4 * 16 * 16).reshape(4, 16, 16) % 250).astype(np.uint8)


def _write(path, **scales):
    with create_czi(str(path), exist_ok=True) as writer:
        for z in range(VOLUME.shape[0]):
            writer.write(data=VOLUME[z][..., None], plane={"Z": z, "C": 0, "T": 0}, location=(0, 0))
        writer.write_metadata(channel_names={0: "c0"}, **scales)
    return str(path)


def test_routes_to_czi_reader_and_reads_values_and_axes(tmp_path):
    reader = Readers.auto_detect(_write(tmp_path / "a.czi", scale_x=8e-9, scale_y=8e-9, scale_z=40e-9))
    assert isinstance(reader, CZIReader)
    store = reader.get_tensorstore()
    assert tuple(store.domain.labels) == ("z", "y", "x")
    assert np.array_equal(store.read().result(), VOLUME)


def test_stated_scale_is_returned_in_nm(tmp_path):
    reader = CZIReader(_write(tmp_path / "b.czi", scale_x=8e-9, scale_y=8e-9, scale_z=40e-9))
    assert dict(reader.get_voxel_sizes()) == pytest.approx({"x": 8.0, "y": 8.0, "z": 40.0})
    assert reader.has_voxel_metadata() is True


def test_file_without_scale_is_not_calibrated(tmp_path):
    assert CZIReader(_write(tmp_path / "c.czi")).has_voxel_metadata() is False
