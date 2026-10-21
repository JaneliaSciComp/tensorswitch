"""Crop one bbox of a label array and convert it to a Paintera N5 with the Java helper `paintera-convert` (hand-off to proofreaders).

The helper has no crop option and reads position from the source array blindly, in X,Y,Z, whatever the array's own axis order is
(mia_annotation#1138: a z,y,x offset swapped Z and X and was invisible on isotropic data). So this module
  1. reads the label array's geometry BY AXIS NAME (the parent group's OME multiscales: names, scale and translation in nm),
  2. writes the bbox as a small standalone zarr3 array whose flat `offset`/`voxel_size` block is X,Y,Z,
  3. runs the helper with `--dataset-resolution` and `--dataset-offset` given explicitly, in X,Y,Z,
  4. checks the finished N5 against an offset computed independently from the geometry (per axis), not against the input.
Nothing here reimplements Paintera's index structures; the helper writes them.
"""
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field

DEFAULT_HELPER = "/groups/scicompsoft/home/chend/tools/paintera-conversion-helper"
MIN_HELPER = (1, 3, 1)                       # --slice-positions and the to-paintera fixes need 1.3.1 or newer
BLOCK = (32, 32, 32)
SCALES = ((2, 2, 2), (2, 2, 2))              # three levels, s0-s2 (the JJ procedure)


class PainteraConvertError(Exception):
    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


@dataclass
class Geometry:
    """A label array's place in the world, by axis name (nm)."""
    array_dir: str
    axes: list                                # storage order, e.g. ['z', 'y', 'x']
    shape: list
    dtype: str
    scale_nm: dict = field(default_factory=dict)
    translation_nm: dict = field(default_factory=dict)


def _read(path):
    with open(path) as f:
        return json.load(f)


def read_geometry(array_dir):
    """Axes, shape, dtype, scale and translation of `array_dir` (an `s0`-like array) from its parent group's OME multiscales.
    The array's own flat attributes are never used: in the LSD sources they are raster units."""
    array_dir = os.path.abspath(array_dir.rstrip("/"))
    arr = _read(os.path.join(array_dir, "zarr.json"))
    if arr.get("node_type") != "array":
        raise PainteraConvertError("source_not_an_array", f"{array_dir} is not a zarr3 array")
    group_dir, name = os.path.split(array_dir)
    gj = os.path.join(group_dir, "zarr.json")
    if not os.path.isfile(gj):
        raise PainteraConvertError("geometry_unknown", f"{group_dir} has no zarr.json with OME multiscales")
    ms = (_read(gj).get("attributes", {}).get("ome", {}).get("multiscales") or [None])[0]
    if not ms:
        raise PainteraConvertError("geometry_unknown", f"{group_dir} has no OME multiscales")
    axes = [a["name"] for a in ms["axes"]]
    if sorted(axes) != ["x", "y", "z"] or len(arr["shape"]) != 3:
        raise PainteraConvertError("geometry_unknown", f"expected a 3D z/y/x array, got axes {axes} and shape {arr['shape']}")
    ds = next((d for d in ms["datasets"] if d["path"] == name), None)
    if ds is None:
        raise PainteraConvertError("geometry_unknown", f"dataset {name!r} is not in {group_dir}'s multiscales")
    scale = next((t["scale"] for t in ds["coordinateTransformations"] if t["type"] == "scale"), None)
    trans = next((t["translation"] for t in ds["coordinateTransformations"] if t["type"] == "translation"), [0.0] * 3)
    if scale is None:
        raise PainteraConvertError("geometry_unknown", f"dataset {name!r} has no scale")
    return Geometry(array_dir, axes, list(arr["shape"]), arr["data_type"],
                    dict(zip(axes, map(float, scale))), dict(zip(axes, map(float, trans))))


def expected_geometry_xyz(geom, start_by_axis):
    """(resolution, offset) of the crop starting at `start_by_axis` {axis: voxel}, both X,Y,Z in nm. Computed only from the
    source's own scale and translation: base translation plus scale times the start index, per axis name."""
    res = [geom.scale_nm[a] for a in "xyz"]
    off = [geom.translation_nm[a] + geom.scale_nm[a] * start_by_axis[a] for a in "xyz"]
    return res, off


def _bbox_by_axis(geom, start_zyx, size_zyx):
    start = dict(zip("zyx", start_zyx))
    size = dict(zip("zyx", size_zyx))
    for a in "zyx":
        i = geom.axes.index(a)
        if start[a] < 0 or size[a] <= 0 or start[a] + size[a] > geom.shape[i]:
            raise PainteraConvertError("bbox_overflow", f"the bbox {a}:{start[a]}-{start[a] + size[a]} does not fit the array (size {geom.shape[i]}); nothing was clipped")
    return start, size


def crop_to_source(geom, start_zyx, size_zyx, out_dir, group=None, chunk=128):
    """Write the bbox (given z,y,x) as a standalone zarr3 array `out_dir/segments/s0`, its flat block X,Y,Z. Returns (max_id, n_nonzero_voxels)."""
    import numpy as np
    import tensorstore as ts
    start, size = _bbox_by_axis(geom, start_zyx, size_zyx)
    src = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": geom.array_dir}}).result()
    shape = [size[a] for a in geom.axes]
    arr_dir = os.path.join(out_dir, "segments", "s0")
    os.makedirs(arr_dir, exist_ok=True)
    dst = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": arr_dir},
                   "metadata": {"shape": shape, "data_type": geom.dtype,
                                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [min(chunk, n) for n in shape]}},
                                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                                           {"name": "zstd", "configuration": {"level": 3}}]},
                   "create": True, "delete_existing": False}).result()
    sl = tuple(slice(start[a], start[a] + size[a]) for a in geom.axes)
    lead = max(1, (64 * 1024 * 1024) // max(1, int(np.prod(shape[1:])) * 8))          # slabs along the first stored axis
    max_id, nonzero = 0, 0
    for z0 in range(0, shape[0], lead):
        z1 = min(z0 + lead, shape[0])
        part = src[(slice(sl[0].start + z0, sl[0].start + z1),) + sl[1:]].read().result()
        dst[z0:z1].write(part).result()
        max_id = max(max_id, int(part.max()) if part.size else 0)
        nonzero += int(np.count_nonzero(part))
    res, off = expected_geometry_xyz(geom, start)
    attrs = {"axis_names": ["d0", "d1", "d2"], "units": ["", "", ""], "voxel_size": res, "offset": off, "types": ["space", "space", "space"]}
    zj = os.path.join(arr_dir, "zarr.json")
    meta = _read(zj)
    meta["attributes"] = attrs                                    # X,Y,Z, independent of the stored axis order
    with open(zj, "w") as f:
        json.dump(meta, f, indent=2)
    root = {"zarr_format": 3, "node_type": "group", "attributes": {"ome": {"version": "0.5", "multiscales": [{
        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in geom.axes],
        "datasets": [{"path": "segments", "coordinateTransformations": [
            {"type": "scale", "scale": [geom.scale_nm[a] for a in geom.axes]},
            {"type": "translation", "translation": [geom.translation_nm[a] + geom.scale_nm[a] * start[a] for a in geom.axes]}]}]}],
        "image-label": {}}}}
    with open(os.path.join(out_dir, "zarr.json"), "w") as f:
        json.dump(root, f, indent=2)
    with open(os.path.join(out_dir, "segments", "zarr.json"), "w") as f:
        json.dump({"zarr_format": 3, "node_type": "group", "attributes": root["attributes"]}, f, indent=2)
    return max_id, nonzero


def helper_version(helper_dir):
    """The highest (1, 3, 1)-style version among the helper's jars, or None."""
    lib = os.path.join(helper_dir, "target", "dependency")
    found = []
    for n in os.listdir(lib) if os.path.isdir(lib) else []:
        m = re.match(r"paintera-conversion-helper-(\d+)\.(\d+)\.(\d+)", n)
        if m and not n.endswith(("-sources.jar", "-tests.jar", "-javadoc.jar")):
            found.append(tuple(int(g) for g in m.groups()))
    return max(found) if found else None


def check_helper(helper_dir=DEFAULT_HELPER, script=None):
    """The helper script path; raises when Java or the helper is missing or older than 1.3.1."""
    explicit = script is not None
    script = script or os.path.join(helper_dir, "startup-scripts", "paintera-convert.sh")
    if shutil.which("java") is None and not os.environ.get("JAVA_HOME") and not explicit:
        raise PainteraConvertError("java_missing", "no `java` on PATH and no JAVA_HOME")
    if not os.path.isfile(script):
        raise PainteraConvertError("helper_missing", f"{script} not found")
    ver = MIN_HELPER if explicit else helper_version(helper_dir)         # an explicit script (a test double, a site wrapper) has no version to read
    if ver is None or ver < MIN_HELPER:
        raise PainteraConvertError("helper_too_old", f"paintera-conversion-helper {ver} found, {MIN_HELPER} or newer is needed")
    return script


def build_command(script, source_dir, out_container, resolution_xyz, offset_xyz, block=BLOCK, scales=SCALES, spark_master="local[4]"):
    """`paintera-convert to-paintera` for the cropped source. LabelMultisetType (no winner-takes-all), X,Y,Z resolution and offset given explicitly."""
    fmt = lambda v: ",".join(repr(float(x)) for x in v)         # noqa: E731
    return [script, "to-paintera", f"--container={source_dir}", "-d", "segments/s0", "--type=label",
            f"--output-container={out_container}", "--target-dataset=labels",
            "--scale", *[",".join(str(s) for s in sc) for sc in scales],
            f"--block-size={','.join(str(b) for b in block)}",
            f"--dataset-resolution={fmt(resolution_xyz)}", f"--dataset-offset={fmt(offset_xyz)}",
            f"--spark-master={spark_master}"]


def check_output(out_container, expected_res_xyz, expected_off_xyz, expected_max_id=None, levels=3, tol=1e-6):
    """[{name, status, detail}] for the finished N5. The offset and resolution are compared with values computed from the source's
    geometry per axis (X,Y,Z), so a swapped axis fails even on near-isotropic data."""
    out = []

    def add(name, ok, detail=""):
        out.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})

    lab = os.path.join(out_container, "labels")
    try:
        a = _read(os.path.join(lab, "attributes.json"))
        d = _read(os.path.join(lab, "data", "attributes.json"))
    except (OSError, ValueError) as e:
        add("paintera_metadata", False, f"cannot read labels/attributes.json or labels/data/attributes.json: {e}")
        return out
    add("paintera_metadata", (a.get("painteraData") or {}).get("type") == "label" and "labelBlockLookup" in a,
        f"painteraData {a.get('painteraData')}, labelBlockLookup {'present' if 'labelBlockLookup' in a else 'missing'}")
    missing = [n for n in ("unique-labels", "label-to-block-mapping") if not os.path.isdir(os.path.join(lab, n))]     # fragment-segment-assignment is created by Paintera on first open
    add("paintera_structures", not missing, f"missing {missing}" if missing else "unique-labels and label-to-block-mapping present")
    have = sorted(n for n in os.listdir(os.path.join(lab, "data")) if re.fullmatch(r"s\d+", n))
    add("levels", have == [f"s{i}" for i in range(levels)], f"levels {have}, expected s0-s{levels - 1}")
    off, res = d.get("offset"), d.get("resolution")
    bad = [a_ for a_, got, want in zip("xyz", off or [], expected_off_xyz) if abs(got - want) > tol * max(1.0, abs(want))]
    add("offset_xyz", off is not None and len(off) == 3 and not bad,
        f"N5 offset {off} (X,Y,Z), expected {[round(v, 6) for v in expected_off_xyz]}" + (f"; differs on {bad}" if bad else ""))
    badr = [a_ for a_, got, want in zip("xyz", res or [], expected_res_xyz) if abs(got - want) > tol * max(1.0, abs(want))]
    add("resolution_xyz", res is not None and len(res) == 3 and not badr,
        f"N5 resolution {res} (X,Y,Z), expected {[round(v, 6) for v in expected_res_xyz]}" + (f"; differs on {badr}" if badr else ""))
    if expected_max_id is not None:
        add("max_id", a.get("maxId") == expected_max_id, f"maxId {a.get('maxId')}, the crop's largest id is {expected_max_id}")
    return out


def set_handoff_permissions(out_container, group="miaai"):
    """Paintera edits the N5 in place, so unlike /data/ it needs group write: folders 2775 (setgid), files 664, group `group`."""
    import grp
    gid = grp.getgrnam(group).gr_gid
    for root, dirs, files in os.walk(out_container):
        for n in dirs:
            p = os.path.join(root, n)
            os.chown(p, -1, gid)
            os.chmod(p, 0o2775)
        for n in files:
            p = os.path.join(root, n)
            os.chown(p, -1, gid)
            os.chmod(p, 0o664)
    os.chown(out_container, -1, gid)
    os.chmod(out_container, 0o2775)


def convert_bbox(array_dir, start_zyx, size_zyx, out_container, workdir, group="miaai", helper_dir=DEFAULT_HELPER, script=None,
                 runner=None, keep_source=False, spark_master="local[4]"):
    """Crop the bbox, convert it, check it. Returns {ok, checks, max_id, nonzero_voxels, offset_xyz, resolution_xyz, command, log}.
    Never overwrites: an existing `out_container` is an error. The temporary source is removed unless `keep_source`."""
    script = check_helper(helper_dir, script)
    if os.path.exists(out_container):
        raise PainteraConvertError("output_exists", f"{out_container} already exists; it is never overwritten")
    geom = read_geometry(array_dir)
    start, _ = _bbox_by_axis(geom, start_zyx, size_zyx)
    res, off = expected_geometry_xyz(geom, start)
    os.makedirs(workdir, exist_ok=True)
    source = os.path.join(workdir, "source.zarr")
    if os.path.exists(source):
        raise PainteraConvertError("workdir_not_clean", f"{source} already exists")
    max_id, nonzero = crop_to_source(geom, start_zyx, size_zyx, source, group)
    os.makedirs(out_container)
    if group:
        shutil.chown(out_container, group=group)
        os.chmod(out_container, 0o2775)                          # pre-created setgid, so the helper's last-written files inherit the group
    cmd = build_command(script, source, out_container, res, off, spark_master=spark_master)
    log = os.path.join(workdir, "paintera_convert.log")
    runner = runner or _run
    rc = runner(cmd, log)
    if rc != 0:
        raise PainteraConvertError("helper_failed", f"paintera-convert exited with {rc}; see {log}")
    if group:
        set_handoff_permissions(out_container, group)
    checks = check_output(out_container, res, off, max_id)
    if not keep_source:
        shutil.rmtree(source, ignore_errors=True)
    return {"ok": all(c["status"] == "pass" for c in checks), "checks": checks, "max_id": max_id, "nonzero_voxels": nonzero,
            "offset_xyz": off, "resolution_xyz": res, "command": cmd, "log": log}


def _run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False).returncode
