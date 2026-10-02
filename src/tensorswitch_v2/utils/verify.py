"""
Check a converted OME-Zarr container against what was asked for and, above all,
against the source data.

Exit codes of jobs say little: a job can finish "DONE" having written nothing, and a
transposed axis, a dropped slice or a wrong voxel size pass every metadata-only check.
``verify_output`` therefore compares the numbers: when the source is available, the
written array must be identical to it (the whole array if it is small, otherwise
slices spread through it).

Each check is ``pass``, ``fail`` or ``unverified`` (it could not run: the reason is
given). The overall result is ``fail`` if any check failed, ``unverified`` if none failed
but a required check could not run, else ``pass``. ``unverified`` is never shown as a pass.
Nothing is ever deleted or modified, except writing ``verification.json`` into the container.
"""

import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

FULL_COMPARE_BYTES = 1024 ** 3        # compare the whole array up to this size
SLAB_BYTES = 256 * 1024 ** 2          # read this much at a time when comparing everything
REPORT_NAME = "verification.json"
_JSON_NAMES = ("zarr.json", ".zattrs")


# ----------------------------------------------------------------------------- reading metadata

def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _group_attrs(group_dir: str) -> dict:
    """OME attributes of a zarr group (zarr3: zarr.json attributes.ome, zarr2: .zattrs)."""
    meta = _read_json(os.path.join(group_dir, "zarr.json"))
    if meta is not None:
        attrs = meta.get("attributes", {})
        return attrs.get("ome", attrs)
    return _read_json(os.path.join(group_dir, ".zattrs")) or {}


def _multiscale(group_dir: str) -> Optional[dict]:
    scales = _group_attrs(group_dir).get("multiscales")
    return scales[0] if scales else None


def find_layers(output: str) -> Dict[str, List[str]]:
    """Image and label groups of a container: {'images': ['raw'], 'labels': ['segmentation']}."""
    images = []
    for name in sorted(os.listdir(output)):
        path = os.path.join(output, name)
        if os.path.isdir(path) and name not in ("labels", "OME") and not name.endswith(".tmp") \
                and _multiscale(path):
            images.append(name)
    labels_dir = os.path.join(output, "labels")
    labels = []
    if os.path.isdir(labels_dir):
        labels = [n for n in sorted(os.listdir(labels_dir))
                  if os.path.isdir(os.path.join(labels_dir, n)) and not n.endswith(".tmp")
                  and _multiscale(os.path.join(labels_dir, n))]
    return {"images": images, "labels": labels}


def _voxel_nm(ms: dict, level: int = 0) -> Dict[str, float]:
    """Spatial scale of a multiscale level in nanometers, by axis name."""
    from .format_loaders import convert_to_nanometers

    axes = ms.get("axes", [])
    scale = None
    for transform in ms["datasets"][level].get("coordinateTransformations", []):
        if transform.get("type") == "scale":
            scale = transform["scale"]
    if scale is None:
        return {}
    return {a["name"].lower(): convert_to_nanometers(scale[i], a.get("unit", "nanometer"))
            for i, a in enumerate(axes) if a.get("type") == "space" and i < len(scale)}


def _open(path: str):
    from ..api import Readers

    return Readers.auto_detect(path).get_tensorstore()


def _parse_voxel(value) -> Dict[str, float]:
    if not value:
        return {}
    if isinstance(value, str):
        parts = [float(v) for v in value.replace(" ", "").split(",")]
        return dict(zip("xyz", parts))
    return {k.lower(): float(v) for k, v in value.items() if v}


# ----------------------------------------------------------------------------- identity

def _slab_indices(n: int, samples: int) -> List[int]:
    chosen = {0, n - 1, n // 2}
    rng = np.random.default_rng(0)
    while len(chosen) < min(n, max(samples, 3)):
        chosen.add(int(rng.integers(0, n)))
    return sorted(chosen)


def compare_with_source(out_store, src_store, bbox=None, bbox_axes=None, samples: int = 5,
                        full_bytes: int = FULL_COMPARE_BYTES) -> Dict[str, Any]:
    """Is the written array identical to the source (optionally inside a bbox)?

    Returns {'status': 'pass'|'fail'|'unverified', 'detail': str, 'mode': 'full'|'sampled'}.
    Only dimensions of length 1 may differ between the two (squeezed axes).
    """
    src_shape, out_shape = [int(n) for n in src_store.shape], [int(n) for n in out_store.shape]
    origin = [0] * len(src_shape)
    region = list(src_shape)
    if bbox:
        b_origin, b_size = bbox
        covered = list(bbox_axes) if bbox_axes else list(range(len(src_shape) - len(b_size), len(src_shape)))
        if len(covered) != len(b_size) or any(not 0 <= i < len(src_shape) for i in covered):
            return {"status": "unverified", "detail": "bbox does not fit the source axes", "mode": None}
        for i, o, n in zip(covered, b_origin, b_size):
            origin[i], region[i] = int(o), int(n)

    src_axes = [i for i, n in enumerate(region) if n != 1]
    out_axes = [i for i, n in enumerate(out_shape) if n != 1]
    if [region[i] for i in src_axes] != [out_shape[i] for i in out_axes]:
        return {"status": "unverified", "mode": None,
                "detail": f"cannot map the source region {tuple(region)} onto the output {tuple(out_shape)} "
                          f"(axes reordered or resized?)"}
    if np.dtype(src_store.dtype.numpy_dtype) != np.dtype(out_store.dtype.numpy_dtype):
        return {"status": "fail", "mode": None,
                "detail": f"dtype changed: source {src_store.dtype.numpy_dtype}, output {out_store.dtype.numpy_dtype}"}
    if not src_axes:                      # a single value
        src_axes, out_axes = [0], [0]

    def read(store, axis, lo, hi, base):
        index = []
        for d, n in enumerate(store.shape):
            if d == axis:
                index.append(slice(base[d] + lo, base[d] + hi))
            elif store is src_store:
                index.append(slice(origin[d], origin[d] + region[d]))
            else:
                index.append(slice(0, int(n)))
        return np.squeeze(np.asarray(store[tuple(index)].read().result()))

    s_ax, o_ax = src_axes[0], out_axes[0]
    n = region[s_ax]
    itemsize = np.dtype(out_store.dtype.numpy_dtype).itemsize
    total = int(np.prod(region)) * itemsize
    row_bytes = max(1, total // max(n, 1))
    if total <= full_bytes:
        step = max(1, SLAB_BYTES // row_bytes)
        spans, mode = [(i, min(i + step, n)) for i in range(0, n, step)], "full"
    else:
        spans, mode = [(i, i + 1) for i in _slab_indices(n, samples)], "sampled"

    zero = [0] * len(out_shape)
    for lo, hi in spans:
        a = read(src_store, s_ax, lo, hi, origin)   # source slab sits at the bbox origin
        b = read(out_store, o_ax, lo, hi, zero)
        if a.shape != b.shape or not np.array_equal(a, b):
            differing = np.argwhere(a != b)[:1].tolist() if a.shape == b.shape else "shape"
            return {"status": "fail", "mode": mode,
                    "detail": f"output differs from the source in slab {lo}:{hi} along axis {o_ax} "
                              f"(first differing position {differing})"}
    where = "the whole array" if mode == "full" else f"{len(spans)} slices ({', '.join(str(s[0]) for s in spans)})"
    return {"status": "pass", "mode": mode, "detail": f"identical to the source in {where}"}


# ----------------------------------------------------------------------------- the report

def _check(checks: list, name: str, status: str, detail: str):
    checks.append({"name": name, "status": status, "detail": detail})


def _sample_slices(store, count: int = 3) -> List[np.ndarray]:
    shape = [int(n) for n in store.shape]
    axis = next((i for i, n in enumerate(shape) if n > 1), 0)
    out = []
    for idx in sorted({0, shape[axis] // 2, shape[axis] - 1})[:count]:
        index = tuple(idx if d == axis else slice(None) for d in range(len(shape)))
        out.append(np.asarray(store[index].read().result()))
    return out


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("tensorswitch")
    except Exception:
        return "unknown"


def verify_output(output: str, source: Optional[str] = None, expected: Optional[Dict[str, Any]] = None,
                  samples: int = 5, write_report: bool = True) -> Dict[str, Any]:
    """Verify a converted container. See the module docstring for the meaning of the results.

    Args:
        output: the .zarr container.
        source: the file the image was converted from (identity check). Without it the
            result is at best ``unverified``.
        expected: optional dict with ``voxel_size`` ("x,y,z" or {x,y,z} in nm), ``labels``
            ({name: source file}), ``bbox`` ("origin..,size.."), ``bbox_axes``, ``dataset_path``
            (HDF5 source), ``output_dtype`` (values intentionally cast), ``group`` (Unix group
            the files should have), ``image_key``.
        samples: slices compared when the array is too big to compare whole.
        write_report: write ``verification.json`` into the container.
    """
    expected = dict(expected or {})
    checks: List[Dict[str, str]] = []
    report: Dict[str, Any] = {"output": output, "source": source, "checks": checks,
                              "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tensorswitch": _version()}

    if not os.path.isdir(output):
        _check(checks, "structure", "fail", f"output does not exist: {output}")
        return _finish(report, output, write_report=False)

    layers = find_layers(output)
    image_key = expected.get("image_key", "raw")
    wanted = {"images": [image_key] if image_key else [], "labels": list(expected.get("labels") or {})}
    problems = []
    for kind in ("images", "labels"):
        for name in wanted[kind]:
            if name not in layers[kind]:
                problems.append(f"missing {kind[:-1]} '{name}'")
    leftovers = sorted(os.path.relpath(os.path.join(d, n), output)
                       for d, dirs, files in os.walk(output) for n in dirs + files if n.endswith((".tmp", ".part")))
    if leftovers:
        problems.append(f"leftover temporary paths: {leftovers[:3]}")
    if not layers["images"] and not layers["labels"]:
        problems.append("no OME-Zarr image or label group found")
    _check(checks, "structure", "fail" if problems else "pass",
           "; ".join(problems) if problems else f"images {layers['images']}, labels {layers['labels']}, no temporary paths")

    all_groups = [(n, os.path.join(output, n)) for n in layers["images"]] + \
                 [(f"labels/{n}", os.path.join(output, "labels", n)) for n in layers["labels"]]
    stores: Dict[str, Any] = {}
    for label, group in all_groups:
        ms = _multiscale(group)
        issues, shapes = [], []
        for ds in ms["datasets"]:
            level_dir = os.path.join(group, ds["path"])
            try:
                shapes.append((ds["path"], _open(level_dir)))
            except Exception as err:
                issues.append(f"level {ds['path']} does not open ({type(err).__name__})")
        if shapes:
            stores[label] = shapes[0][1]
            base = [int(n) for n in shapes[0][1].shape]
            base_scale = _voxel_nm(ms, 0)
            axes = [a["name"].lower() for a in ms["axes"]]
            for k, (path, store) in enumerate(shapes[1:], 1):
                if store.dtype != shapes[0][1].dtype:
                    issues.append(f"level {path} has a different dtype")
                level_scale = _voxel_nm(ms, k)
                for d, axis in enumerate(axes):
                    if axis in level_scale and axis in base_scale and base_scale[axis] > 0:
                        want = math.ceil(base[d] / (level_scale[axis] / base_scale[axis]))
                        if abs(int(store.shape[d]) - want) > 1:
                            issues.append(f"level {path} axis {axis}: shape {int(store.shape[d])}, expected about {want}")
        _check(checks, f"levels:{label}", "fail" if issues else "pass",
               "; ".join(issues) if issues else f"{len(ms['datasets'])} level(s), consistent")

    # metadata
    want_voxel = _parse_voxel(expected.get("voxel_size"))
    for label in [n for n in layers["images"]][:1]:
        ms = _multiscale(os.path.join(output, label))
        found = _voxel_nm(ms, 0)
        if not found:
            _check(checks, "voxel_size", "fail", "the output states no voxel size")
        elif not want_voxel:
            _check(checks, "voxel_size", "unverified",
                   f"output states {found}; no expected voxel size was given to compare")
        else:
            bad = {a: (want_voxel[a], found.get(a)) for a in want_voxel
                   if a in found and not math.isclose(want_voxel[a], found[a], rel_tol=1e-6)}
            missing = [a for a in want_voxel if a not in found]
            _check(checks, "voxel_size", "fail" if bad or missing else "pass",
                   f"expected vs found (nm): {bad}" if bad else
                   f"missing axes {missing}" if missing else f"matches the expected {want_voxel}")

    # data: not constant
    for label, store in stores.items():
        try:
            slices = _sample_slices(store)
        except Exception as err:
            _check(checks, f"data:{label}", "fail", f"cannot read the array ({type(err).__name__}: {str(err)[:80]})")
            continue
        constant = all(s.size and s.min() == s.max() for s in slices)
        is_label = label.startswith("labels/")
        _check(checks, f"data:{label}", "fail" if constant and not is_label else "pass",
               "sampled slices are all one value" + ("" if not is_label else " (allowed for labels)") if constant
               else "sampled slices contain data")

    # identity with the source
    from ..__main__ import parse_bbox, parse_bbox_axes

    bbox = parse_bbox(expected["bbox"]) if expected.get("bbox") else None
    bbox_axes = parse_bbox_axes(expected["bbox_axes"]) if expected.get("bbox_axes") else None
    to_compare = []
    if layers["images"] and image_key in stores:
        to_compare.append((image_key, source, stores[image_key]))
    for name, path in (expected.get("labels") or {}).items():
        if f"labels/{name}" in stores:
            to_compare.append((f"labels/{name}", path, stores[f"labels/{name}"]))
    if not to_compare:
        to_compare = [(next(iter(stores), "output"), source, None)]
    for label, path, out_store in to_compare:
        name = f"identity:{label}"
        dataset = expected.get("dataset_path") if label == image_key else None
        if path and "::" in path:                    # "file.h5::volumes/labels/x": a dataset inside the file
            path, dataset = path.split("::", 1)
        if not path:
            _check(checks, name, "unverified", "no source file was given, so the data could not be compared")
        elif out_store is None:
            _check(checks, name, "unverified", "the output array is not available")
        elif expected.get("output_dtype"):
            _check(checks, name, "unverified", "values were intentionally cast (output_dtype); not compared")
        elif not os.path.exists(path):
            _check(checks, name, "unverified", f"source is no longer available: {path}")
        else:
            try:
                from ..api import Readers

                reader = (Readers.hdf5(path, dataset_path=dataset)
                          if dataset and path.lower().endswith((".h5", ".hdf5", ".hdf", ".he5"))
                          else Readers.auto_detect(path))
                result = compare_with_source(out_store, reader.get_tensorstore(), bbox=bbox,
                                             bbox_axes=bbox_axes, samples=samples)
                _check(checks, name, result["status"], result["detail"])
            except Exception as err:
                _check(checks, name, "unverified", f"source could not be read ({type(err).__name__}: {str(err)[:80]})")

    # group ownership
    group = expected.get("group")
    if group:
        import grp

        try:
            gid = grp.getgrnam(group).gr_gid
            wrong = [os.path.relpath(os.path.join(d, n), output) for d, dirs, files in os.walk(output)
                     for n in dirs + files if os.lstat(os.path.join(d, n)).st_gid != gid]
            _check(checks, "group", "fail" if wrong else "pass",
                   f"{len(wrong)} entries not in group {group}, e.g. {wrong[:2]}" if wrong else f"all entries in group {group}")
        except KeyError:
            _check(checks, "group", "unverified", f"no Unix group named {group!r}")
    return _finish(report, output, write_report)


def _finish(report: Dict[str, Any], output: str, write_report: bool) -> Dict[str, Any]:
    checks = report["checks"]
    failed = [c["name"] for c in checks if c["status"] == "fail"]
    unverified = [c["name"] for c in checks if c["status"] == "unverified"]
    report["failures"], report["unverified"] = failed, unverified
    report["overall"] = "fail" if failed else "unverified" if unverified else "pass"
    if write_report:
        try:
            with open(os.path.join(output, REPORT_NAME), "w") as handle:
                json.dump(report, handle, indent=2)
        except OSError as err:
            report["report_written"] = False
            report["report_error"] = str(err)
    return report
