"""A level 0 that sits at an offset keeps it in every pyramid level's translation (plus the half-voxel shift)."""
import json
import os

from tensorswitch_v2.utils.metadata_utils import (_compute_translation, update_ome_multiscale_metadata,
                                                  update_ome_multiscale_metadata_zarr2)


def test_compute_translation_adds_the_base_translation():
    assert _compute_translation([18.0] * 3, [9.0] * 3, "mode", [7515.0, 12825.0, 3879.0]) == [7519.5, 12829.5, 3883.5]
    assert _compute_translation([36.0] * 3, [9.0] * 3, "mean", [7515.0, 12825.0, 3879.0]) == [7528.5, 12838.5, 3892.5]
    assert _compute_translation([18.0] * 3, [9.0] * 3, "stride", [5.0, 6.0, 7.0]) == [5.0, 6.0, 7.0]
    assert _compute_translation([18.0] * 3, [9.0] * 3, "mode") == [4.5, 4.5, 4.5]               # no base translation: as before


def _level(path, shape, factor, v):
    os.makedirs(os.path.join(path, v), exist_ok=True)
    json.dump({"shape": shape, "attributes": {"tensorswitch": {"cumulative_factor": [factor] * 3}}},
              open(os.path.join(path, v, "zarr.json"), "w"))


def test_zarr3_levels_keep_the_crops_translation(tmp_path):
    root = str(tmp_path)
    meta = {"attributes": {"ome": {"multiscales": [{
        "axes": [{"name": n, "type": "space", "unit": "nanometer"} for n in "xyz"],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": [9.0, 9.0, 9.0]}, {"type": "translation", "translation": [7515.0, 12825.0, 3879.0]}]}]}]}}}
    json.dump(meta, open(os.path.join(root, "zarr.json"), "w"))
    _level(root, [100, 100, 100], 1, "s0"); _level(root, [50, 50, 50], 2, "s1"); _level(root, [25, 25, 25], 4, "s2")
    update_ome_multiscale_metadata(root, max_level=2, prefix="s", downsample_method="mode")
    ds = json.load(open(os.path.join(root, "zarr.json")))["attributes"]["ome"]["multiscales"][0]["datasets"]
    assert ds[0]["coordinateTransformations"][1]["translation"] == [7515.0, 12825.0, 3879.0]
    assert ds[1]["coordinateTransformations"][1]["translation"] == [7519.5, 12829.5, 3883.5]
    assert ds[2]["coordinateTransformations"][1]["translation"] == [7528.5, 12838.5, 3892.5]
