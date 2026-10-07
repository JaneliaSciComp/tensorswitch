"""
Turn a dataset-catalog record (mia-agentic-search YAML) into a conversion plan.

The planner only reads the record. It never downloads or converts anything: it
returns the fetch and convert calls to run, and warnings for everything it could
not decide from the record. Nothing is guessed: a missing voxel size, a missing
file URL or an unsupported format becomes a warning, not a default.

``plan_record(record, output_dir)`` is pure. ``load_record(ref)`` does the I/O
(local path, GitHub URL or record id).
"""

import copy
import fnmatch
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

CATALOG_REPO = "AI-HHMI/mia-agentic-search"
IN_SCOPE_DIMENSIONS = ("3D", "3D+t")
MCP_LIMIT_BYTES = 2 * 1024 ** 3
CLUSTER_LIMIT_BYTES = 50 * 1024 ** 3
BACKGROUND_FETCH_BYTES = 200 * 1024 ** 2   # slow servers deliver a few hundred MB in more than a tool call allows
# Formats whose header does not give a size TensorSwitch can trust (HDF5 has none, BioSR-style MRC files
# store micrometers in the angstrom field, NIfTI units are unreliable). Without a voxel size in the record
# the converter refuses these, so the planner does not plan them.
HEADER_NOT_TRUSTED = {"hdf5", "mrc", "nifti"}
SHORT_FIRST_AXIS = 10     # TensorSwitch guesses axis names for HDF5; a first dimension this short becomes 'c'

# record array format -> how TensorSwitch can read it
STORE_FORMATS = {"zarr", "ome-zarr", "n5", "precomputed"}   # folders: readable remotely, not fetchable
CONVERTIBLE = {"tiff", "ome-tiff", "hdf5", "mrc", "nifti", "czi", "nd2", "zarr", "ome-zarr", "n5", "precomputed"}
NEEDS_SLICE_FOLDER = {"png", "jpeg"}      # one file per slice: needs the whole folder
TABLE_FORMATS = {"csv", "json"}           # annotations stored as tables, not arrays
EXTENSIONS = {
    "tiff": (".tif", ".tiff"), "ome-tiff": (".tif", ".tiff"), "hdf5": (".h5", ".hdf5", ".hdf", ".he5"),
    "mrc": (".mrc", ".mrcs", ".rec", ".ali", ".st", ".map", ".rec.nad"), "nifti": (".nii", ".nii.gz"), "czi": (".czi",),
    "nd2": (".nd2",), "png": (".png",), "jpeg": (".jpg", ".jpeg"), "csv": (".csv",), "json": (".json",),
    "zarr": (".zarr",), "ome-zarr": (".zarr",), "n5": (".n5",),
}
# words that name a file's role, not the sample it belongs to (dropped when pairing raw with labels)
_ROLE_WORDS = {"image", "images", "img", "raw", "data", "input", "inputs", "label", "labels", "mask", "masks", "seg",
               "segmentation", "gt", "groundtruth", "target", "targets", "annotation", "annotations"}
_GENERIC_TOKENS = {"data", "the", "and", "set", "tif", "tiff", "png", "h5", "hdf5", "mrc", "nii", "gz", "zip",
                   "file", "files", "nnn", "name", "stem", "uid", "id"}


class RecordError(Exception):
    """The record could not be loaded or is not a usable record."""


# ----------------------------------------------------------------------------- loading

def _github_raw(url: str) -> str:
    """blob/tree URLs of the catalog -> raw.githubusercontent URLs."""
    m = re.match(r"https://github\.com/([^/]+/[^/]+)/blob/(.+)", url)
    return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}" if m else url


def _resolve_id(record_id: str, ref: str = "main") -> str:
    import requests

    resp = requests.get(f"https://api.github.com/repos/{CATALOG_REPO}/git/trees/{ref}?recursive=1", timeout=60)
    if resp.status_code != 200:
        raise RecordError(f"could not list the catalog (HTTP {resp.status_code}); pass a file path or URL instead")
    hits = [t["path"] for t in resp.json().get("tree", [])
            if t["path"].startswith("datasets/") and t["path"].endswith(f"/{record_id}.yaml")]
    if not hits:
        raise RecordError(f"no record with id {record_id!r} in {CATALOG_REPO}")
    return f"https://raw.githubusercontent.com/{CATALOG_REPO}/{ref}/{hits[0]}"


def load_record(ref: str) -> Dict[str, Any]:
    """Load a record from a local .yaml path, a GitHub URL, or a catalog record id."""
    import yaml

    ref = ref.strip()
    if os.path.isfile(ref):
        text = open(ref, encoding="utf-8").read()
    else:
        import requests

        if urlsplit(ref).scheme in ("http", "https"):
            url = _github_raw(ref)
        elif re.fullmatch(r"[A-Za-z0-9._-]+", ref):
            url = _resolve_id(ref)
        else:
            raise RecordError(f"not a file, URL or record id: {ref!r}")
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            raise RecordError(f"HTTP {resp.status_code} for {url}")
        text = resp.text
    try:
        record = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise RecordError(f"not valid YAML: {err}") from err
    if not isinstance(record, dict) or "id" not in record or "imaging" not in record:
        raise RecordError("not a catalog record (needs at least 'id' and 'imaging')")
    return record


# ----------------------------------------------------------------------------- helpers

def _tokens(text: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 1 and t not in _GENERIC_TOKENS
            and not t.isdigit()}


def _split_spec(spec: str) -> Tuple[str, Optional[str]]:
    if "::" in spec:
        url, member = spec.split("::", 1)
        return url, member
    return spec, None


def _name_of(spec: str) -> str:
    url, member = _split_spec(spec)
    return member or unquote(os.path.basename(urlsplit(url).path))


def _ext_ok(spec: str, fmt: str) -> bool:
    exts = EXTENSIONS.get(fmt)
    return True if exts is None else _name_of(spec).lower().endswith(exts)


def _expand_braces(pattern: str) -> List[str]:
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    return [r for alt in m.group(1).split(",")
            for r in _expand_braces(pattern[:m.start()] + alt + pattern[m.end():])]


def _split_dataset(pattern: str, fmt: Optional[str]) -> Tuple[str, Optional[str]]:
    """HDF5 patterns often end with the dataset name: 'a/*.h5 (volumes/raw)' or 'a.h5 :: FOV0'."""
    if fmt != "hdf5":
        return pattern, None
    m = re.match(r"^(?P<glob>.*?\.(?:h5|hdf5|hdf))\s*(?:\((?P<a>[^)]*)\)|::\s*(?P<b>\S+))\s*$", pattern)
    if not m:
        return pattern, None
    dataset = (m.group("a") or m.group("b") or "").strip()
    return m.group("glob"), (dataset if dataset and "<" not in dataset else None)


def _glob_variants(pattern: str, fmt: Optional[str]) -> List[str]:
    """Pattern -> fnmatch globs: drop the dataset suffix, expand {a,b}, turn <x> and NNN/PP placeholders into *."""
    glob, _ = _split_dataset(pattern, fmt)
    glob = re.sub(r"<[^>]*>", "*", glob)
    glob = re.sub(r"N{2,}|P{2,}", "*", glob)
    variants = _expand_braces(glob)
    variants += [v.split("::", 1)[-1].strip() for v in variants if "::" in v]
    return variants


def _candidates(spec: str) -> List[str]:
    """Names a sample file can be matched under: member, 'zipname/member', URL path."""
    url, member = _split_spec(spec)
    path = unquote(urlsplit(url).path).lstrip("/")
    if member:
        return [member, f"{os.path.basename(path)}/{member}", f"{path}/{member}"]
    return [os.path.basename(path), path]


def _match_score(spec: str, array: Dict[str, Any]) -> int:
    """How well a concrete sample file fits an array entry (0 = does not fit)."""
    fmt = array.get("format")
    if fmt and not _ext_ok(spec, fmt):
        return 0
    pattern = array.get("path_pattern") or ""
    names = _candidates(spec)
    if pattern and any(fnmatch.fnmatch(n, g) or fnmatch.fnmatch(n, "*" + g)
                       for g in _glob_variants(pattern, fmt) for n in names):
        return 100
    return len(_tokens(pattern) & _tokens(" ".join(names)))


def _assign_samples(specs: List[str], arrays: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    """Give each sample file to the array it fits best; misfits and fuzzy ties stay unassigned.

    Arrays that share an exact pattern (e.g. raw and label datasets inside one HDF5
    file) all get the file.
    """
    assigned: Dict[int, List[str]] = {}
    for spec in specs:
        scores = [(_match_score(spec, a), i) for i, a in enumerate(arrays)]
        best = max(s for s, _ in scores) if scores else 0
        winners = [i for s, i in scores if s == best]
        if best > 0 and (len(winners) == 1 or best == 100):
            for winner in winners:
                assigned.setdefault(winner, []).append(spec)
    return assigned


def _voxel_arg(record: Dict[str, Any], dimensionality: str) -> Tuple[Optional[str], List[str]]:
    vox = (record.get("imaging") or {}).get("voxel_size_nm") or {}
    x, y, z = vox.get("x"), vox.get("y"), vox.get("z")
    if x and y and z:
        return f"{x:g},{y:g},{z:g}", []
    if x and y:
        return None, ["record gives only x and y voxel size (no z); 3D conversion needs z"]
    return None, ["record has no voxel size (imaging.voxel_size_nm); conversion will use the file header "
                  "and is refused if the header is incomplete"]


def _tiff_input_axes(array: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """(input_axes, problem) for a plain TIFF array. The record's `axes` name every axis, slowest to fastest.

    A TIFF read without them gets the reader's own names (`i` for pages, `s` for samples per pixel), which
    are neither z nor a channel, so the record has to say what the axes are.
    """
    axes = (array.get("axes") or "").strip().lower()
    if not axes:
        return None, ("the record gives no `axes` for this TIFF array; without them TensorSwitch would name the "
                      "pages `i` and samples per pixel `s` (not z or a channel), so it is not planned until "
                      "technical.arrays[].axes is filled in")
    if len(set(axes)) != len(axes) or set(axes) - set("tcszyx"):
        return None, f"the record's axes {axes!r} are not a string of distinct letters from t, c, s, z, y, x"
    shape = array.get("shape")
    if shape and len(shape) != len(axes):
        return None, f"the record's axes {axes!r} name {len(axes)} axes but its shape has {len(shape)}"
    return axes, None


def _local_path(source_dir: str, spec: str) -> str:
    """Where fetch_dataset puts a spec inside source_dir (mirrors utils.fetch)."""
    url, member = _split_spec(spec)
    name = member if member else unquote(os.path.basename(urlsplit(url).path))
    return os.path.join(source_dir, *name.replace("\\", "/").split("/"))


def _label_source(args: Dict[str, Any]) -> str:
    """Source of a label for verify_output: the file, plus '::dataset' inside an HDF5 file."""
    path = args["input_path"]
    if args.get("dataset_path") and path.lower().endswith(EXTENSIONS["hdf5"]):
        return f"{path}::{args['dataset_path']}"
    return path


def _fmt_gb(n: Optional[int]) -> str:
    return "unknown size" if not n else f"{n / 1024 ** 3:.1f} GB"


# ----------------------------------------------------------------------------- planner

def plan_record(record: Dict[str, Any], output_dir: str, project: Optional[str] = None) -> Dict[str, Any]:
    """Build a conversion plan for one record (no network, no files touched).

    Args:
        record: parsed catalog record.
        output_dir: folder for the converted containers; staged downloads go to
            ``<output_dir>/source/`` (delete after verification).
        project: LSF project, needed in the steps for datasets over the in-process limit.
    """
    imaging, data = record.get("imaging") or {}, record.get("data") or {}
    technical = record.get("technical") or {}
    dimensionality = imaging.get("dimensionality", "unknown")
    record_id = record["id"]
    warnings: List[str] = []
    plan: Dict[str, Any] = {
        "record": {"id": record_id, "title": record.get("title"), "dimensionality": dimensionality,
                   "repository": record.get("repository"), "license": (record.get("license") or {}).get("spdx"),
                   "size": _fmt_gb(data.get("size_bytes"))},
        "status": "ready", "arrays": [], "steps": [], "warnings": warnings,
    }

    if dimensionality not in IN_SCOPE_DIMENSIONS:
        plan["status"] = "blocked"
        warnings.append(f"{dimensionality} records are out of scope for now (3D and 3D+t only; independent 2D "
                        f"collections are deferred)")
        return plan
    if dimensionality == "3D+t":
        warnings.append("3D+t record: the time axis is not checked by the planner yet")
    if data.get("access") not in (None, "open"):
        warnings.append(f"access is {data.get('access')!r}: files may need a login or a request first")

    size = data.get("size_bytes")
    if size and size > CLUSTER_LIMIT_BYTES:
        warnings.append(f"whole dataset is {_fmt_gb(size)} (over 50 GB); only the sample unit is planned")
    voxel, voxel_warnings = _voxel_arg(record, dimensionality)
    warnings.extend(voxel_warnings)

    arrays = technical.get("arrays") or []
    sample_specs = list((technical.get("sample") or {}).get("urls") or [])
    if not arrays:
        plan["status"] = "blocked"
        warnings.append("record has no technical.arrays (the enricher has not described the files); "
                        "nothing to plan")
        return plan
    if not sample_specs:
        plan["status"] = "blocked"
        warnings.append("record has no technical.sample.urls and download_url is a "
                        f"{'landing page or folder' if data.get('download_url') else 'missing value'}; "
                        "there is no concrete file to fetch")
        return plan

    assigned = _assign_samples(sample_specs, arrays)
    matched = {s for specs in assigned.values() for s in specs}
    for spec in sample_specs:
        if spec not in matched:
            warnings.append(f"sample file not matched to any array (ignored): {_name_of(spec)}")

    source_dir = os.path.join(output_dir, "source")
    containers: Dict[str, Dict[str, Any]] = {}
    label_count = 0
    main_raw = next((i for i, a in enumerate(arrays)
                     if a.get("role") == "raw" and a.get("format") in CONVERTIBLE and assigned.get(i)), None)

    for index, array in enumerate(arrays):
        fmt, role = array.get("format"), array.get("role")
        entry: Dict[str, Any] = {"index": index, "role": role, "format": fmt, "axes": array.get("axes"),
                                 "dtype": array.get("dtype"), "files": assigned.get(index, []),
                                 "convertible": False, "notes": []}
        plan["arrays"].append(entry)
        notes = entry["notes"]
        if fmt in TABLE_FORMATS or (fmt == "other" and role == "label"):
            notes.append("annotation is a table or custom file, not an array; not converted")
            continue
        if fmt in NEEDS_SLICE_FOLDER:
            notes.append(f"{fmt} slices need the whole folder fetched as one stack; not supported by the planner yet")
            continue
        if fmt not in CONVERTIBLE:
            notes.append(f"format {fmt!r} is not supported by TensorSwitch yet")
            continue
        if not entry["files"]:
            notes.append("no sample file matched this array")
            continue
        remote_store = fmt in STORE_FORMATS and "::" not in entry["files"][0]
        if fmt in STORE_FORMATS and not remote_store:
            notes.append(f"{fmt} stores are folders; a store inside a zip cannot be fetched as one member yet")
            continue
        if role not in ("raw", "label", "target"):
            notes.append(f"role {role!r} is not converted")
            continue
        if array.get("shape_varies"):
            notes.append("shape varies between files; only the sample file is planned")
        axes, shape = array.get("axes"), array.get("shape")
        if role in ("raw", "label", "target") and ((axes and "z" not in axes.lower()) or (shape and len(shape) == 2)):
            notes.append(f"the record is {dimensionality} but this array is described as 2D "
                         f"(axes {axes!r}, shape {shape}); the sample file is probably a single slice, not a volume")
        if array.get("axes") and array["axes"] not in ("zyx", "czyx", "tzyx", "tczyx"):
            notes.append(f"axes {array['axes']!r} are unusual; check the output axes after converting")
        if role == "label" and (array.get("dtype") or "").startswith("float"):
            notes.append(f"label array has dtype {array['dtype']}; labels are normally integers")
        if array.get("alignment") in ("scaled", "offset", "cropped", "transform-provided"):
            notes.append(f"alignment with the raw data is {array['alignment']!r}; the record's voxel size may not "
                         f"apply to this array")
        if not voxel and fmt in HEADER_NOT_TRUSTED:
            notes.append(f"the record has no complete voxel size and {fmt} headers are not reliable, so the "
                         f"converter would refuse; not planned until imaging.voxel_size_nm is filled in")
            entry["needs"] = "voxel_size"
            continue
        tiff_axes = None
        if fmt == "tiff":
            tiff_axes, problem = _tiff_input_axes(array)
            if problem:
                notes.append(problem)
                entry["needs"] = "axes"
                continue
        dataset_path = _split_dataset(array.get("path_pattern") or "", fmt)[1]
        if fmt == "hdf5" and not dataset_path:
            notes.append("HDF5 dataset name is not in the record; TensorSwitch picks the main dataset itself "
                         "(a common name such as 'raw' or 'volume', else the largest)")
        entry["convertible"] = True

        # which container does this array go to?
        if role == "raw" and index == main_raw:
            key = "main"
        elif role == "label":
            key = "main" if main_raw is not None else "labels_only"
        else:
            key = f"{role}_{index}"
        container = containers.setdefault(key, {"path": os.path.join(
            output_dir, f"{record_id}.zarr" if key == "main" else f"{record_id}_{key}.zarr"), "has_base": False})

        spec = entry["files"][0]
        if len(entry["files"]) > 1:
            notes.append(f"{len(entry['files'])} sample files matched; the first is planned")
        args: Dict[str, Any] = {"input_path": spec if remote_store else _local_path(source_dir, spec),
                                "output_path": container["path"]}
        if voxel:
            args["voxel_size"] = voxel
        if tiff_axes:
            args["input_axes"] = tiff_axes
        if fmt == "hdf5":
            args["dataset_path"] = dataset_path
            shape = array.get("shape")
            if shape and len(shape) == 3 and 0 < int(shape[0]) <= SHORT_FIRST_AXIS:
                # HDF5 has no axis names; TensorSwitch reads a first dimension this short as a
                # channel, so a 3-D record's z-stack would be written without a z axis
                args["relabel_axis"] = "c=z"
                notes.append(f"first axis has only {shape[0]} entries, which TensorSwitch would read as a channel "
                             f"for HDF5; the step passes relabel_axis='c=z' so it becomes z")
        if role == "label":
            label_count += 1
            args.update({"is_label": True, "data_type": "labels", "label_key": "segmentation" if label_count == 1
                         else f"segmentation_{label_count}"})
            if key == "main" or container["has_base"]:
                args["add_to_existing"] = True
        entry["convert_args"] = args
        container["has_base"] = True
        entry["fetch"] = None if remote_store else {"spec": spec, "dest_dir": source_dir}

    convertible = [a for a in plan["arrays"] if a["convertible"]]
    needs = sorted({a["needs"] for a in plan["arrays"] if a.get("needs")})
    if needs:
        plan["needs"] = needs
    if not convertible:
        plan["status"] = "blocked"
        warnings.append("no array could be planned" + (f" (the record must provide: {', '.join(needs)})" if needs else ""))
        return plan
    if len(convertible) < len(plan["arrays"]):
        plan["status"] = "partial"

    # raw before label so add_to_existing finds its container
    ordered = sorted(convertible, key=lambda a: (a["role"] == "label", a["index"]))
    sample_size = (technical.get("sample") or {}).get("size_bytes")
    use_cluster = bool(sample_size and sample_size > MCP_LIMIT_BYTES)
    if use_cluster and not project:
        warnings.append("sample is over 2 GB: pass an LSF project so the plan can use submit_job")
    if sample_size and sample_size > BACKGROUND_FETCH_BYTES:
        warnings.append(f"sample is {_fmt_gb(sample_size)}: run fetch_dataset with background=True and call it again "
                        f"until it reports success, before the convert steps")
    fetched: List[Dict[str, Any]] = []
    for entry in ordered:
        if entry["fetch"] and entry["fetch"] not in fetched:
            fetched.append(entry["fetch"])
            plan["steps"].append({"tool": "fetch_dataset", "args": dict(entry["fetch"])})
        args = dict(entry["convert_args"])
        if use_cluster:
            args["project"] = project or "<LSF project>"
        plan["steps"].append({"tool": "submit_job" if use_cluster else "convert", "args": args})
    for container in containers.values():
        members = [a["convert_args"] for a in ordered if a["convert_args"]["output_path"] == container["path"]]
        images = [a for a in members if not a.get("is_label")]
        labels = [a for a in members if a.get("is_label")]

        verify_args: Dict[str, Any] = {"output_path": container["path"],
                                       "source_path": images[0]["input_path"] if images else "",
                                       "image_key": "raw" if images else ""}
        if images and images[0].get("dataset_path"):
            verify_args["dataset_path"] = images[0]["dataset_path"]
        if members and members[0].get("voxel_size"):
            verify_args["voxel_size"] = members[0]["voxel_size"]
        if labels:
            verify_args["labels"] = ";".join(f"{a['label_key']}={_label_source(a)}" for a in labels)
        plan["steps"].append({"tool": "verify_output", "args": verify_args})
    unnamed: Dict[str, int] = {}
    for a in ordered:
        if a["format"] == "hdf5" and a["convert_args"].get("dataset_path") is None:
            unnamed[a["convert_args"]["input_path"]] = unnamed.get(a["convert_args"]["input_path"], 0) + 1
    if any(n > 1 for n in unnamed.values()):
        warnings.append("several arrays use the same HDF5 file with dataset_path=null, so TensorSwitch would pick "
                        "the same dataset for each: run inspect_dataset on the fetched file and fill them in")
    return plan


# ----------------------------------------------------------------------------- whole dataset

def _zip_url(record: Dict[str, Any]) -> Optional[str]:
    for spec in (record.get("technical") or {}).get("sample", {}).get("urls") or []:
        if "::" in spec:
            return spec.split("::", 1)[0]
    url = (record.get("data") or {}).get("download_url") or ""
    return url if urlsplit(url).path.lower().endswith(".zip") else None


def _stem_tokens(filename: str) -> Tuple[str, ...]:
    stem = filename
    for _ in range(2):
        stem, ext = os.path.splitext(stem)
        if ext.lower() not in (".gz", ".tif", ".tiff", ".nii", ".h5", ".hdf5", ".mrc", ".rec", ".zarr", ".n5"):
            stem += ext
            break
    tokens = [t for t in re.split(r"[^a-z0-9]+", stem.lower()) if t]
    if tokens and re.fullmatch(r"[a-z]\d+", tokens[0]):          # X02 / Y02 -> 02
        tokens[0] = tokens[0][1:]
    return tuple(t for t in tokens if t not in _ROLE_WORDS)


def _pair_key(member: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Identity of the sample a file belongs to: its folders and file name without role words."""
    *folders, filename = member.split("/")
    kept = tuple(f.lower() for f in folders
                 if not {t for t in re.split(r"[^a-z0-9]+", f.lower()) if t} <= _ROLE_WORDS)
    return kept, _stem_tokens(filename)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def plan_dataset(record: Dict[str, Any], output_dir: str, project: Optional[str] = None,
                 lister=None) -> Dict[str, Any]:
    """Plan the conversion of every file the record describes, one container per sample.

    The files come from the central directory of the record's zip download. Raw and label
    files are paired by their folders and file name once role words ("images", "masks",
    "input", "labels") and a leading X/Y letter are removed; a file with no partner gets its
    own container and is reported, never guessed. Datasets over 50 GB are skipped and listed.
    Needs the network (zip listing); ``lister(url)`` can be injected for tests.
    """
    if lister is None:
        from .fetch import list_zip as lister
    technical = record.get("technical") or {}
    arrays = technical.get("arrays") or []
    base = plan_record(record, output_dir, project)
    plan: Dict[str, Any] = {"record": base["record"], "scope": "dataset", "status": "ready", "samples": [],
                            "unpaired": [], "skipped": [], "steps": [], "warnings": []}
    warnings = plan["warnings"]
    if base["status"] == "blocked" and not arrays:
        plan["status"], plan["warnings"] = "blocked", base["warnings"]
        return plan
    zip_url = _zip_url(record)
    if not zip_url:
        plan["status"] = "blocked"
        warnings.append("whole-dataset planning needs a zip download to list its files; this record has none "
                        "(the sample-unit plan still works)")
        return plan
    try:
        entries = lister(zip_url)
    except Exception as err:
        plan["status"] = "blocked"
        warnings.append(f"could not list the files of {zip_url}: {err}")
        return plan

    by_array: Dict[int, Dict[Any, Tuple[str, int]]] = {i: {} for i in range(len(arrays))}
    for entry in entries:
        spec = f"{zip_url}::{entry.name}"
        for index, array in enumerate(arrays):
            if _match_score(spec, array) == 100:
                key = _pair_key(entry.name)
                if array.get("role") in ("label", "target") and array.get("alignment") != "same-grid":
                    # never pair on the name alone: the record has to say the label sits on the raw's grid
                    key = (key[0], key[1] + (f"unpaired{index}",))
                    plan["unpaired"].append(
                        f"{entry.name}: the record gives alignment {array.get('alignment')!r}, not 'same-grid', "
                        f"so it is not paired with a raw file")
                if key in by_array[index]:
                    plan["unpaired"].append(f"{entry.name}: same sample key as {by_array[index][key][0]}")
                else:
                    by_array[index][key] = (entry.name, entry.size)
    total = sum(size for members in by_array.values() for _, size in members.values())
    plan["files"] = {"matched": sum(len(m) for m in by_array.values()), "bytes_uncompressed": total}
    if total > CLUSTER_LIMIT_BYTES:
        plan["status"] = "skipped"
        plan["skipped"].append({"record": record["id"], "bytes": total})
        warnings.append(f"whole dataset is {_fmt_gb(total)} uncompressed, over 50 GB: skipped")
        return plan
    if not total:
        plan["status"] = "blocked"
        warnings.append("no file in the zip matches the record's path_pattern entries")
        return plan

    keys = sorted({k for members in by_array.values() for k in members})
    shared = os.path.commonprefix([list(k[0]) for k in keys]) if keys else []
    used = set()
    for key in keys:
        members = {i: by_array[i][key] for i in by_array if key in by_array[i]}
        first = members[min(members)][0]
        name = _safe_name("_".join(list(key[0][len(shared):]) + [os.path.splitext(os.path.basename(first))[0]]))
        while name in used:
            name += "_2"
        used.add(name)
        if len(members) < len(arrays):
            missing = [arrays[i].get("role") for i in range(len(arrays)) if i not in members]
            plan["unpaired"].append(f"{first}: no {'/'.join(map(str, missing))} partner found")
        sample = copy.deepcopy(record)
        sample["id"] = f"{record['id']}_{name}"
        sample["technical"]["arrays"] = [{k: v for k, v in a.items() if k not in ("shape", "shape_varies")}
                                         for a in arrays]
        sample["technical"]["sample"] = {"urls": [f"{zip_url}::{m[0]}" for m in members.values()],
                                         "size_bytes": sum(m[1] for m in members.values())}
        sub = plan_record(sample, output_dir, project)
        plan["samples"].append({"name": name, "files": [m[0] for m in members.values()], "status": sub["status"],
                                "warnings": sub["warnings"], "container_id": sample["id"]})
        plan["steps"].extend(sub["steps"])
        if sub["status"] != "ready":
            plan["status"] = "partial"
    if plan["unpaired"]:
        warnings.append(f"{len(plan['unpaired'])} files have no partner or a clashing key; see 'unpaired'")
    plan["cleanup"] = os.path.join(output_dir, "source")
    if not plan["steps"]:
        plan["status"] = "blocked"
    return plan
