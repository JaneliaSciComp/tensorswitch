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
import re
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


def _clean_names(labels) -> Optional[List[str]]:
    """Axis names from a TensorStore domain, lowercased; None when any is missing or repeated."""
    names = [str(n).lower().replace("channel", "c") for n in labels or []]
    return names if names and all(names) and len(set(names)) == len(names) else None


def compare_with_source(out_store, src_store, bbox=None, bbox_axes=None, samples: int = 5,
                        full_bytes: int = FULL_COMPARE_BYTES, src_names=None, out_names=None) -> Dict[str, Any]:
    """Is the written array identical to the source (optionally inside a bbox)?

    Returns {'status': 'pass'|'fail'|'unverified', 'detail': str, 'mode': 'full'|'sampled'}.
    With ``src_names`` and ``out_names`` (axis names of the source and the output) the arrays are matched
    axis by axis by name, so a source whose axes were renamed or reordered (non-spatial axes moved first)
    can be compared; axes present on one side only must have length 1. Without names the axes are matched
    by position and length, and only dimensions of length 1 may differ (squeezed axes).
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

    by_name = bool(src_names and out_names and len(src_names) == len(src_shape)
                   and len(out_names) == len(out_shape)
                   and len(set(src_names)) == len(src_names) and len(set(out_names)) == len(out_names))
    if by_name:
        common = [n for n in out_names if n in src_names]
        extra = [(n, region[src_names.index(n)]) for n in src_names if n not in out_names] + \
                [(n, out_shape[out_names.index(n)]) for n in out_names if n not in src_names]
        bad_extra = [n for n, size in extra if size != 1]
        if not common or bad_extra or any(region[src_names.index(n)] != out_shape[out_names.index(n)] for n in common):
            return {"status": "unverified", "mode": None,
                    "detail": f"cannot match the source axes {src_names} {tuple(region)} to the output axes "
                              f"{out_names} {tuple(out_shape)} by name"}
        slab_name = next((n for n in common if out_shape[out_names.index(n)] != 1), common[0])
        s_ax, o_ax = src_names.index(slab_name), out_names.index(slab_name)
    else:
        src_axes = [i for i, n in enumerate(region) if n != 1]
        out_axes = [i for i, n in enumerate(out_shape) if n != 1]
        if [region[i] for i in src_axes] != [out_shape[i] for i in out_axes]:
            return {"status": "unverified", "mode": None,
                    "detail": f"cannot map the source region {tuple(region)} onto the output {tuple(out_shape)} "
                              f"(axes reordered or resized?)"}
        if not src_axes:                      # a single value
            src_axes, out_axes = [0], [0]
        s_ax, o_ax = src_axes[0], out_axes[0]
    if np.dtype(src_store.dtype.numpy_dtype) != np.dtype(out_store.dtype.numpy_dtype):
        return {"status": "fail", "mode": None,
                "detail": f"dtype changed: source {src_store.dtype.numpy_dtype}, output {out_store.dtype.numpy_dtype}"}

    def read(store, axis, lo, hi, base):
        index = []
        for d, n in enumerate(store.shape):
            if d == axis:
                index.append(slice(base[d] + lo, base[d] + hi))
            elif store is src_store:
                index.append(slice(origin[d], origin[d] + region[d]))
            else:
                index.append(slice(0, int(n)))
        return np.asarray(store[tuple(index)].read().result())

    def aligned(a, b):
        """Source slab and output slab with the same axes in the same order (by name, or squeezed)."""
        if by_name:
            drop_a = tuple(i for i, n in enumerate(src_names) if n not in common)
            drop_b = tuple(i for i, n in enumerate(out_names) if n not in common)
            a = np.squeeze(a, axis=drop_a) if drop_a else a
            b = np.squeeze(b, axis=drop_b) if drop_b else b
            kept_src = [n for n in src_names if n in common]
            kept_out = [n for n in out_names if n in common]
            return np.transpose(a, [kept_src.index(n) for n in kept_out]), b
        return np.squeeze(a), np.squeeze(b)

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
        a, b = aligned(read(src_store, s_ax, lo, hi, origin),     # source slab sits at the bbox origin
                       read(out_store, o_ax, lo, hi, zero))
        if a.shape != b.shape or not np.array_equal(a, b):
            differing = np.argwhere(a != b)[:1].tolist() if a.shape == b.shape else "shape"
            return {"status": "fail", "mode": mode,
                    "detail": f"output differs from the source in slab {lo}:{hi} along axis {o_ax} "
                              f"(first differing position {differing})"}
    where = "the whole array" if mode == "full" else f"{len(spans)} slices ({', '.join(str(s[0]) for s in spans)})"
    how = " (axes matched by name)" if by_name else ""
    return {"status": "pass", "mode": mode, "detail": f"identical to the source in {where}{how}"}


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


def _levels_on_disk(group_dir: str) -> set:
    return {n for n in os.listdir(group_dir) if re.fullmatch(r"s\d+", n) and os.path.isdir(os.path.join(group_dir, n))}


def _check_listed_levels(checks: list, output: str, groups: list):
    """Every resolution level on disk is listed in its group's metadata, and in the root's for the image groups.

    A pyramid can be written in full while the metadata still lists fewer levels (a viewer then shows only those),
    most often in the root zarr.json when several pyramid jobs ran at the same time.
    """
    problems = []
    for label, group in groups:
        ms = _multiscale(group)
        on_disk = _levels_on_disk(group)
        listed = {ds["path"] for ds in ms["datasets"]}
        if on_disk - listed:
            problems.append(f"{label}: levels {sorted(on_disk - listed)} exist but are not listed in its zarr.json")
        if listed - on_disk:
            problems.append(f"{label}: levels {sorted(listed - on_disk)} are listed but missing on disk")
    root = _multiscale(output)
    if root:
        by_group: Dict[str, set] = {}
        for ds in root["datasets"]:
            head, _, level = ds["path"].rpartition("/")
            by_group.setdefault(head, set()).add(level)
        for head, listed in by_group.items():
            group = os.path.join(output, head) if head else output
            if not os.path.isdir(group):
                problems.append(f"root zarr.json lists '{head}', which does not exist")
                continue
            on_disk = _levels_on_disk(group)
            if on_disk != listed:
                problems.append(f"root zarr.json lists {sorted(listed)} for '{head or '.'}' but the folder has "
                                f"{sorted(on_disk)}")
    _check(checks, "levels_listed", "fail" if problems else "pass",
           "; ".join(problems) if problems else "every level on disk is listed in its metadata and in the root")


def _check_zarr2_leftovers(checks: list, output: str):
    """A zarr3 container should not hold zarr2 metadata files (.zgroup, .zattrs, .zarray)."""
    if not os.path.exists(os.path.join(output, "zarr.json")):
        return
    found = sorted(os.path.relpath(os.path.join(d, n), output)
                   for d, dirs, files in os.walk(output) for n in files if n in (".zgroup", ".zattrs", ".zarray"))
    _check(checks, "no_zarr2_files", "fail" if found else "pass",
           f"zarr2 metadata inside a zarr3 container: {found[:3]}" if found else "no zarr2 metadata files")


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
            the files should have), ``image_key``, ``input_axes`` (axis names of the image source, one letter
            per axis, as passed to the conversion) and ``label_input_axes`` ({label name: axes}).
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
    out_names: Dict[str, List[str]] = {}
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
            out_names[label] = [str(a["name"]).lower() for a in ms["axes"]]
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

    _check_listed_levels(checks, output, all_groups)
    _check_zarr2_leftovers(checks, output)

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
            reader = None
            try:
                from ..api import Readers

                reader = (Readers.hdf5(path, dataset_path=dataset)
                          if dataset and path.lower().endswith((".h5", ".hdf5", ".hdf", ".he5"))
                          else Readers.auto_detect(path))
                src_store = reader.get_tensorstore()
                given = expected.get("input_axes") if label == image_key else \
                    (expected.get("label_input_axes") or {}).get(label.split("/", 1)[-1])
                src_names = _clean_names(src_store.domain.labels)
                if given:
                    given = str(given).strip().lower()
                    if len(given) != len(src_store.shape):
                        raise ValueError(f"input_axes {given!r} names {len(given)} axes but the source has "
                                         f"{len(src_store.shape)}")
                    src_names = list(given)
                result = compare_with_source(out_store, src_store, bbox=bbox, bbox_axes=bbox_axes, samples=samples,
                                             src_names=src_names, out_names=out_names.get(label))
                if result["status"] == "unverified" and src_names:
                    # names did not line up (for instance the output was relabeled and verify was not told):
                    # try the old match by position and length before giving up
                    plain = compare_with_source(out_store, src_store, bbox=bbox, bbox_axes=bbox_axes, samples=samples)
                    if plain["status"] != "unverified":
                        result = plain
                _check(checks, name, result["status"], result["detail"])
            except Exception as err:
                _check(checks, name, "unverified", f"source could not be read ({type(err).__name__}: {str(err)[:80]})")
            finally:
                # readers keep the source file open until collected; a caller that deletes the
                # source next (e.g. on NFS) would otherwise leave a hidden placeholder file behind
                reader = result = src_store = None
                import gc
                gc.collect()

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
