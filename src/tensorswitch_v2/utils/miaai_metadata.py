"""
Metadata conventions of the ``miaai`` preset, applied after the data is written.

* ``finalize_container``: every multiscales block (root, image groups, label groups) loses its ``name`` and
  gets an outer ``coordinateTransformations`` (identity, or 1/expansion factor on the spatial axes).
* The preset leaves a marker in the root ``zarr.json`` so that later steps that rewrite the metadata, such as a
  pyramid started by a separate job, apply the same conventions without being told again.
* ``merge_extra_attributes``: add caller-supplied attributes (``--extra_attributes``) to the group a conversion
  wrote, without touching ``ome``, ``_software`` or the marker.

Everything is idempotent and zarr3 only. ``python -m tensorswitch_v2.utils.miaai_metadata <container>`` applies it
to an existing container.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

MARKER_KEY = "tensorswitch"
PROTECTED_KEYS = ("ome", "_software", MARKER_KEY)


def _read(path: str) -> Optional[dict]:
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _write(path: str, doc: dict) -> None:
    with open(path, "w") as handle:
        json.dump(doc, handle, indent=2)


def mark_container(container: str, preset: str = "miaai", expansion_factor: Optional[float] = None) -> None:
    """Record in the root zarr.json that this container follows a preset's metadata conventions."""
    path = os.path.join(container, "zarr.json")
    doc = _read(path)
    if doc is None:
        return
    marker = {"preset": preset}
    if expansion_factor:
        marker["expansion_factor"] = expansion_factor
    attributes = doc.setdefault("attributes", {})
    if attributes.get(MARKER_KEY) != marker:
        attributes[MARKER_KEY] = marker
        _write(path, doc)


def read_marker(path: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """(container root, marker) for a container, or for a group inside one (looks up to three folders up)."""
    current = os.path.abspath(path)
    for _ in range(4):
        doc = _read(os.path.join(current, "zarr.json"))
        marker = ((doc or {}).get("attributes") or {}).get(MARKER_KEY)
        if isinstance(marker, dict) and marker.get("preset"):
            return current, marker
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None, {}


def _outer_transform(axes: list, expansion_factor: Optional[float]) -> list:
    factor = 1.0 / expansion_factor if expansion_factor else 1.0
    scale = [round(factor, 12) if a.get("type") == "space" else 1.0 for a in axes]
    return [{"type": "scale", "scale": scale}]


def finalize_container(container: str, expansion_factor: Optional[float] = None) -> list:
    """Apply the conventions to every multiscales block in a zarr3 container. Returns the files that changed."""
    from .verify import find_layers

    if not os.path.exists(os.path.join(container, "zarr.json")):
        return []
    if expansion_factor is None:
        expansion_factor = read_marker(container)[1].get("expansion_factor")
    layers = find_layers(container)
    folders = [container] + [os.path.join(container, n) for n in layers["images"]] \
        + [os.path.join(container, "labels", n) for n in layers["labels"]]
    changed = []
    for folder in folders:
        path = os.path.join(folder, "zarr.json")
        doc = _read(path)
        scales = (((doc or {}).get("attributes") or {}).get("ome") or {}).get("multiscales") or []
        if not scales:
            continue
        before = json.dumps(scales, sort_keys=True)
        ms = scales[0]
        ms.pop("name", None)
        ms["coordinateTransformations"] = _outer_transform(ms.get("axes") or [], expansion_factor)
        if json.dumps(scales, sort_keys=True) != before:
            _write(path, doc)
            changed.append(os.path.relpath(path, container))
    return changed


def apply_marker_if_any(path: str) -> list:
    """Finalize the container that ``path`` belongs to when it carries the marker. No marker, no change."""
    root, marker = read_marker(path)
    if root is None:
        return []
    return finalize_container(root, marker.get("expansion_factor"))


def load_extra_attributes(value: str) -> Dict[str, Any]:
    """``--extra_attributes`` accepts a JSON file or an inline JSON object."""
    value = (value or "").strip()
    if not value:
        return {}
    text = open(value).read() if os.path.isfile(value) else value
    try:
        data = json.loads(text)
    except ValueError as err:
        raise ValueError(f"--extra_attributes is neither a JSON file nor a JSON object: {err}") from err
    if not isinstance(data, dict):
        raise ValueError("--extra_attributes must be a JSON object")
    clash = [k for k in data if k in PROTECTED_KEYS]
    if clash:
        raise ValueError(f"--extra_attributes may not set {clash}: TensorSwitch owns those keys")
    return data


def merge_extra_attributes(group_dir: str, extras: Dict[str, Any]) -> bool:
    """Add top-level attributes to a group's zarr.json; existing ``ome``/``_software`` are kept. True if changed."""
    if not extras:
        return False
    path = os.path.join(group_dir, "zarr.json")
    doc = _read(path)
    if doc is None:
        raise FileNotFoundError(f"cannot add attributes: {path} does not exist")
    attributes = doc.setdefault("attributes", {})
    before = json.dumps(attributes, sort_keys=True)
    for key, value in extras.items():
        attributes[key] = value
    if json.dumps(attributes, sort_keys=True) == before:
        return False
    _write(path, doc)
    return True


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Apply the miaai metadata conventions to an existing zarr3 container.")
    parser.add_argument("container")
    parser.add_argument("--expansion-factor", type=float, default=None,
                        help="expansion microscopy: outer scale becomes 1/factor on the spatial axes")
    args = parser.parse_args(argv)
    mark_container(args.container, "miaai", args.expansion_factor)
    changed = finalize_container(args.container, args.expansion_factor)
    print(f"{len(changed)} file(s) changed" + (": " + ", ".join(changed) if changed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
