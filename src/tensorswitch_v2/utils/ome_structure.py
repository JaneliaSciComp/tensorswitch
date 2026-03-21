"""
OME-NGFF structure management for nested zarr output.

Manages the unified folder structure for image and label data:

    output.zarr/
    ├── zarr.json                 # Root metadata: multiscales + labels list
    ├── raw/                      # Image data
    │   ├── zarr.json             # Image multiscales
    │   └── s0/, s1/...           # Pyramid levels
    └── labels/                   # Labels container
        ├── zarr.json             # Labels list: ["segmentation"]
        └── segmentation/         # Label image
            ├── zarr.json         # Label multiscales + image-label
            └── s0/, s1/...       # Pyramid levels

Usage:
    from tensorswitch_v2.utils.ome_structure import OMEStructure

    # Create structure manager
    ome = OMEStructure('/path/to/output.zarr')

    # Get paths for writing data
    image_path = ome.get_image_data_path()  # /path/to/output.zarr/raw
    label_path = ome.get_label_data_path()  # /path/to/output.zarr/labels/segmentation

    # Write metadata after conversion
    ome.write_all_metadata(image_multiscales, label_multiscales, label_colors)
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class OMEStructureConfig:
    """Configuration for OME-NGFF structure."""
    image_key: str = 'raw'  # Name for image group
    labels_container: str = 'labels'  # Name for labels container
    label_name: str = 'segmentation'  # Name for the label image
    ome_version: str = '0.5'  # OME-NGFF version
    zarr_format: int = 3  # Zarr format version


class OMEStructure:
    """
    Manages OME-NGFF compliant directory structure and metadata.

    This class handles the unified folder structure for both image and
    label data, ensuring proper metadata is written at all required levels.
    """

    def __init__(
        self,
        output_path: str,
        config: Optional[OMEStructureConfig] = None
    ):
        """
        Initialize OME structure manager.

        Args:
            output_path: Root path for output zarr
            config: Optional configuration (uses defaults if None)
        """
        self.output_path = os.path.abspath(output_path)
        self.config = config or OMEStructureConfig()

    # ========================================================================
    # Path Methods
    # ========================================================================

    def get_image_data_path(self) -> str:
        """
        Get path where image data should be written.

        Returns:
            Path to image group: {output}/raw/
        """
        return os.path.join(self.output_path, self.config.image_key)

    def get_label_data_path(self, label_name: Optional[str] = None) -> str:
        """
        Get path where label data should be written.

        Args:
            label_name: Optional label name (uses config default if None)

        Returns:
            Path to label image: {output}/labels/{label_name}/
        """
        name = label_name or self.config.label_name
        return os.path.join(
            self.output_path,
            self.config.labels_container,
            name
        )

    def get_labels_container_path(self) -> str:
        """
        Get path to labels container directory.

        Returns:
            Path to labels container: {output}/labels/
        """
        return os.path.join(self.output_path, self.config.labels_container)

    def get_level_path(self, level: int, is_label: bool = False, label_name: Optional[str] = None) -> str:
        """
        Get path for a specific pyramid level.

        Args:
            level: Pyramid level (0 = highest resolution)
            is_label: True for label data, False for image
            label_name: Optional label name for labels

        Returns:
            Full path to level directory
        """
        if is_label:
            base = self.get_label_data_path(label_name)
        else:
            base = self.get_image_data_path()
        return os.path.join(base, f's{level}')

    # ========================================================================
    # Metadata Path Methods (for downsampling updates)
    # ========================================================================

    def get_metadata_paths_for_image(self) -> List[Dict[str, Any]]:
        """
        Get all metadata paths that need updating when image pyramid changes.

        Returns list of dicts with:
        - path: Full path to zarr.json
        - path_prefix: Prefix for dataset paths in that metadata

        Returns:
            List of metadata path info dicts
        """
        return [
            {
                'path': os.path.join(self.get_image_data_path(), 'zarr.json'),
                'path_prefix': '',  # s0, s1, ...
            },
            {
                'path': os.path.join(self.output_path, 'zarr.json'),
                'path_prefix': self.config.image_key,  # raw/s0, raw/s1, ...
            }
        ]

    def get_metadata_paths_for_labels(self, label_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all metadata paths that need updating when label pyramid changes.

        Returns list of dicts with:
        - path: Full path to zarr.json
        - path_prefix: Prefix for dataset paths in that metadata

        Args:
            label_name: Optional label name (uses config default if None)

        Returns:
            List of metadata path info dicts
        """
        name = label_name or self.config.label_name
        return [
            {
                'path': os.path.join(self.get_label_data_path(name), 'zarr.json'),
                'path_prefix': '',  # s0, s1, ...
            },
            # Note: labels container zarr.json doesn't have multiscales, only labels list
            # Root zarr.json doesn't need per-level updates for labels
        ]

    # ========================================================================
    # Directory Creation
    # ========================================================================

    def create_directory_structure(
        self,
        include_image: bool = True,
        include_labels: bool = False,
        label_name: Optional[str] = None
    ) -> None:
        """
        Create the directory structure for output.

        Args:
            include_image: Create image directory structure
            include_labels: Create labels directory structure
            label_name: Optional label name for labels
        """
        os.makedirs(self.output_path, exist_ok=True)

        if include_image:
            os.makedirs(self.get_image_data_path(), exist_ok=True)

        if include_labels:
            os.makedirs(self.get_label_data_path(label_name), exist_ok=True)

    # ========================================================================
    # Metadata Creation
    # ========================================================================

    def create_base_group_metadata(self) -> Dict:
        """Create base zarr3 group metadata structure."""
        return {
            'zarr_format': self.config.zarr_format,
            'node_type': 'group',
            'attributes': {
                'ome': {
                    'version': self.config.ome_version
                }
            }
        }

    def create_image_multiscales_metadata(
        self,
        axes: List[Dict],
        datasets: List[Dict],
        name: str = 'image',
        data_type: str = 'image'
    ) -> Dict:
        """
        Create multiscales metadata for image data.

        Args:
            axes: List of axis definitions
            datasets: List of dataset entries with path and coordinateTransformations
            name: Name for the multiscales entry
            data_type: Type field ('image' typically)

        Returns:
            Full zarr.json metadata dict
        """
        metadata = self.create_base_group_metadata()
        metadata['attributes']['ome']['multiscales'] = [{
            'axes': axes,
            'datasets': datasets,
            'name': name,
            'type': data_type
        }]
        return metadata

    def create_labels_container_metadata(
        self,
        label_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Create metadata for labels container directory.

        Args:
            label_names: List of label image names (default: [config.label_name])

        Returns:
            zarr.json metadata for labels container
        """
        names = label_names or [self.config.label_name]
        metadata = self.create_base_group_metadata()
        metadata['attributes']['ome']['labels'] = names
        return metadata

    def create_label_image_metadata(
        self,
        axes: List[Dict],
        datasets: List[Dict],
        name: str = 'segmentation',
        colors: Optional[List[Dict]] = None,
        source_image_path: Optional[str] = None,
        num_default_colors: int = 256
    ) -> Dict:
        """
        Create metadata for label image with image-label section.

        Args:
            axes: List of axis definitions
            datasets: List of dataset entries
            name: Name for the multiscales entry
            colors: Optional label colors (generates defaults if None)
            source_image_path: Relative path to source image (e.g., "../../raw")
            num_default_colors: Number of default colors to generate

        Returns:
            zarr.json metadata for label image
        """
        metadata = self.create_base_group_metadata()

        # Add multiscales (no "type": "image" for labels)
        metadata['attributes']['ome']['multiscales'] = [{
            'axes': axes,
            'datasets': datasets,
            'name': name
        }]

        # Generate colors if not provided
        if colors is None:
            from .metadata_utils import generate_default_label_colors
            colors = generate_default_label_colors(num_default_colors)

        # Add image-label section
        image_label = {
            'version': self.config.ome_version,
            'colors': colors
        }

        if source_image_path:
            image_label['source'] = {'image': source_image_path}

        metadata['attributes']['ome']['image-label'] = image_label

        return metadata

    def create_root_metadata(
        self,
        image_multiscales: Optional[List[Dict]] = None,
        has_labels: bool = False,
        image_name: str = 'image'
    ) -> Dict:
        """
        Create root zarr.json metadata.

        Args:
            image_multiscales: Image multiscales data (axes, datasets)
            has_labels: Whether labels directory exists
            image_name: Name for image multiscales

        Returns:
            Root zarr.json metadata
        """
        metadata = self.create_base_group_metadata()

        if image_multiscales:
            # Adjust paths to include image_key prefix
            adjusted_multiscales = self._adjust_dataset_paths(
                image_multiscales,
                self.config.image_key
            )
            metadata['attributes']['ome']['multiscales'] = [{
                'axes': adjusted_multiscales['axes'],
                'datasets': adjusted_multiscales['datasets'],
                'name': image_name,
                'type': 'image'
            }]

        if has_labels:
            metadata['attributes']['ome']['labels'] = [self.config.labels_container]

        return metadata

    def _adjust_dataset_paths(
        self,
        multiscales: Dict,
        prefix: str
    ) -> Dict:
        """
        Adjust dataset paths by adding a prefix.

        Args:
            multiscales: Multiscales dict with 'axes' and 'datasets'
            prefix: Prefix to add to each dataset path

        Returns:
            New multiscales dict with adjusted paths
        """
        adjusted_datasets = []
        for ds in multiscales.get('datasets', []):
            new_ds = ds.copy()
            original_path = ds.get('path', 's0')
            new_ds['path'] = f"{prefix}/{original_path}"
            adjusted_datasets.append(new_ds)

        return {
            'axes': multiscales.get('axes', []),
            'datasets': adjusted_datasets
        }

    # ========================================================================
    # Metadata Writing
    # ========================================================================

    def write_metadata(self, path: str, metadata: Dict) -> None:
        """Write metadata to a zarr.json file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def write_image_metadata(
        self,
        multiscales: Dict,
        name: str = 'image'
    ) -> None:
        """
        Write image metadata to image group zarr.json.

        Args:
            multiscales: Dict with 'axes' and 'datasets'
            name: Name for multiscales entry
        """
        metadata = self.create_image_multiscales_metadata(
            axes=multiscales.get('axes', []),
            datasets=multiscales.get('datasets', []),
            name=name
        )
        path = os.path.join(self.get_image_data_path(), 'zarr.json')
        self.write_metadata(path, metadata)

    def write_labels_container_metadata(
        self,
        label_names: Optional[List[str]] = None
    ) -> None:
        """
        Write labels container metadata.

        Args:
            label_names: List of label image names
        """
        metadata = self.create_labels_container_metadata(label_names)
        path = os.path.join(self.get_labels_container_path(), 'zarr.json')
        self.write_metadata(path, metadata)

    def write_label_image_metadata(
        self,
        multiscales: Dict,
        name: str = 'segmentation',
        colors: Optional[List[Dict]] = None,
        source_image_path: Optional[str] = None,
        label_name: Optional[str] = None
    ) -> None:
        """
        Write label image metadata.

        Args:
            multiscales: Dict with 'axes' and 'datasets'
            name: Name for multiscales entry
            colors: Optional label colors
            source_image_path: Relative path to source image
            label_name: Optional label name for path
        """
        # Default source path if labels and image in same zarr
        if source_image_path is None:
            source_image_path = f"../../{self.config.image_key}"

        metadata = self.create_label_image_metadata(
            axes=multiscales.get('axes', []),
            datasets=multiscales.get('datasets', []),
            name=name,
            colors=colors,
            source_image_path=source_image_path
        )
        path = os.path.join(self.get_label_data_path(label_name), 'zarr.json')
        self.write_metadata(path, metadata)

    def write_root_metadata(
        self,
        image_multiscales: Optional[Dict] = None,
        has_labels: bool = False,
        image_name: str = 'image',
        ome_xml: Optional[str] = None
    ) -> None:
        """
        Write root zarr.json metadata, merging with existing if present.

        Args:
            image_multiscales: Dict with 'axes' and 'datasets' for image
            has_labels: Whether labels directory exists
            image_name: Name for image multiscales
            ome_xml: Raw OME-XML string from source data (stored at attributes.ome_xml)
        """
        path = os.path.join(self.output_path, 'zarr.json')

        # Read existing metadata if present (to preserve image multiscales when adding labels)
        existing_metadata = {}
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    existing_metadata = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_metadata = {}

        # Create new metadata
        metadata = self.create_root_metadata(
            image_multiscales=image_multiscales,
            has_labels=has_labels,
            image_name=image_name
        )

        # Merge: preserve existing multiscales if we're not providing new ones
        existing_ome = existing_metadata.get('attributes', {}).get('ome', {})
        new_ome = metadata.get('attributes', {}).get('ome', {})

        if image_multiscales is None and 'multiscales' in existing_ome:
            new_ome['multiscales'] = existing_ome['multiscales']

        # Merge: preserve existing labels if present and we're adding more
        if 'labels' in existing_ome and 'labels' in new_ome:
            existing_labels = set(existing_ome.get('labels', []))
            new_labels = set(new_ome.get('labels', []))
            new_ome['labels'] = list(existing_labels | new_labels)
        elif 'labels' in existing_ome and 'labels' not in new_ome:
            new_ome['labels'] = existing_ome['labels']

        metadata['attributes']['ome'] = new_ome

        # Preserve or set ome_xml at attributes level (consistent with legacy non-nested mode)
        final_ome_xml = ome_xml or existing_metadata.get('attributes', {}).get('ome_xml')
        if final_ome_xml:
            metadata['attributes']['ome_xml'] = final_ome_xml

        self.write_metadata(path, metadata)

    def write_all_metadata(
        self,
        image_multiscales: Optional[Dict] = None,
        label_multiscales: Optional[Dict] = None,
        image_name: str = 'image',
        label_name: str = 'segmentation',
        label_colors: Optional[List[Dict]] = None
    ) -> None:
        """
        Write all metadata files for the complete structure.

        Args:
            image_multiscales: Image multiscales (axes, datasets) or None
            label_multiscales: Label multiscales (axes, datasets) or None
            image_name: Name for image in metadata
            label_name: Name for label in metadata
            label_colors: Optional label colors
        """
        has_image = image_multiscales is not None
        has_labels = label_multiscales is not None

        # Create directory structure
        self.create_directory_structure(
            include_image=has_image,
            include_labels=has_labels
        )

        # Write image metadata
        if has_image:
            self.write_image_metadata(image_multiscales, name=image_name)

        # Write labels metadata
        if has_labels:
            self.write_labels_container_metadata([self.config.label_name])
            self.write_label_image_metadata(
                multiscales=label_multiscales,
                name=label_name,
                colors=label_colors,
                source_image_path=f"../../{self.config.image_key}" if has_image else None
            )

        # Write root metadata
        self.write_root_metadata(
            image_multiscales=image_multiscales if has_image else None,
            has_labels=has_labels,
            image_name=image_name
        )

    # ========================================================================
    # Metadata Update (for downsampling)
    # ========================================================================

    def add_level_to_image_metadata(
        self,
        level_name: str,
        scale: List[float],
        translation: Optional[List[float]] = None
    ) -> None:
        """
        Add a new pyramid level to image metadata files.

        Updates both the image group zarr.json and root zarr.json.

        Args:
            level_name: Level name (e.g., 's1')
            scale: Scale factors for this level
            translation: Optional translation for this level
        """
        for meta_info in self.get_metadata_paths_for_image():
            self._add_level_to_metadata_file(
                meta_info['path'],
                level_name,
                meta_info['path_prefix'],
                scale,
                translation
            )

    def add_level_to_label_metadata(
        self,
        level_name: str,
        scale: List[float],
        translation: Optional[List[float]] = None,
        label_name: Optional[str] = None
    ) -> None:
        """
        Add a new pyramid level to label metadata files.

        Args:
            level_name: Level name (e.g., 's1')
            scale: Scale factors for this level
            translation: Optional translation for this level
            label_name: Optional label name
        """
        for meta_info in self.get_metadata_paths_for_labels(label_name):
            self._add_level_to_metadata_file(
                meta_info['path'],
                level_name,
                meta_info['path_prefix'],
                scale,
                translation
            )

    def _add_level_to_metadata_file(
        self,
        metadata_path: str,
        level_name: str,
        path_prefix: str,
        scale: List[float],
        translation: Optional[List[float]] = None
    ) -> None:
        """
        Add a level entry to a specific metadata file.

        Args:
            metadata_path: Path to zarr.json file
            level_name: Level name (e.g., 's1')
            path_prefix: Prefix for path (e.g., 'raw' -> 'raw/s1')
            scale: Scale factors
            translation: Optional translation
        """
        if not os.path.exists(metadata_path):
            return

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Find multiscales
        ome = metadata.get('attributes', {}).get('ome', {})
        multiscales = ome.get('multiscales', [])

        if not multiscales:
            return

        # Build dataset path
        dataset_path = f"{path_prefix}/{level_name}" if path_prefix else level_name

        # Create new dataset entry
        new_dataset = {
            'path': dataset_path,
            'coordinateTransformations': [
                {'type': 'scale', 'scale': scale}
            ]
        }

        if translation and any(t != 0 for t in translation):
            new_dataset['coordinateTransformations'].append({
                'type': 'translation',
                'translation': translation
            })

        # Add to first multiscales entry
        multiscales[0]['datasets'].append(new_dataset)

        # Write back
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


# ============================================================================
# Zarr2 OME Structure
# ============================================================================

@dataclass
class OMEStructureZarr2Config:
    """Configuration for Zarr2 OME-NGFF structure."""
    image_key: str = 'raw'  # Name for image group
    labels_container: str = 'labels'  # Name for labels container
    label_name: str = 'segmentation'  # Name for the label image
    ome_version: str = '0.4'  # OME-NGFF version for Zarr2


class OMEStructureZarr2:
    """
    Manages OME-NGFF compliant directory structure and metadata for Zarr2 format.

    Zarr2 uses .zgroup and .zattrs files instead of zarr.json.
    OME metadata is at top level of .zattrs (not nested under attributes.ome).

    Structure:
        output.zarr/
        ├── .zgroup                   # {"zarr_format": 2}
        ├── .zattrs                   # multiscales + labels list
        ├── raw/
        │   ├── .zgroup
        │   ├── .zattrs               # image multiscales
        │   └── 0/, 1/...             # Pyramid levels (numeric for Zarr2)
        └── labels/
            ├── .zgroup
            ├── .zattrs               # {"labels": ["segmentation"]}
            └── segmentation/
                ├── .zgroup
                ├── .zattrs           # multiscales + image-label
                └── 0/, 1/...         # Pyramid levels
    """

    def __init__(
        self,
        output_path: str,
        config: Optional[OMEStructureZarr2Config] = None
    ):
        self.output_path = os.path.abspath(output_path)
        self.config = config or OMEStructureZarr2Config()

    # ========================================================================
    # Path Methods
    # ========================================================================

    def get_image_data_path(self) -> str:
        """Get path where image data should be written."""
        return os.path.join(self.output_path, self.config.image_key)

    def get_label_data_path(self, label_name: Optional[str] = None) -> str:
        """Get path where label data should be written."""
        name = label_name or self.config.label_name
        return os.path.join(
            self.output_path,
            self.config.labels_container,
            name
        )

    def get_labels_container_path(self) -> str:
        """Get path to labels container directory."""
        return os.path.join(self.output_path, self.config.labels_container)

    # ========================================================================
    # Metadata Creation (Zarr2 format)
    # ========================================================================

    def _write_zgroup(self, path: str) -> None:
        """Write .zgroup file."""
        zgroup_path = os.path.join(path, '.zgroup')
        os.makedirs(path, exist_ok=True)
        with open(zgroup_path, 'w') as f:
            json.dump({"zarr_format": 2}, f, indent=2)

    def _write_zattrs(self, path: str, attrs: Dict) -> None:
        """Write .zattrs file."""
        zattrs_path = os.path.join(path, '.zattrs')
        os.makedirs(path, exist_ok=True)
        with open(zattrs_path, 'w') as f:
            json.dump(attrs, f, indent=2)

    def create_image_multiscales_metadata(
        self,
        axes: List[Dict],
        datasets: List[Dict],
        name: str = 'image'
    ) -> Dict:
        """Create multiscales metadata for image data (Zarr2 format)."""
        return {
            "multiscales": [{
                "version": self.config.ome_version,
                "name": name,
                "axes": axes,
                "datasets": datasets
            }]
        }

    def create_labels_container_metadata(
        self,
        label_names: Optional[List[str]] = None
    ) -> Dict:
        """Create metadata for labels container directory."""
        names = label_names or [self.config.label_name]
        return {"labels": names}

    def create_label_image_metadata(
        self,
        axes: List[Dict],
        datasets: List[Dict],
        name: str = 'segmentation',
        colors: Optional[List[Dict]] = None,
        source_image_path: Optional[str] = None,
        num_default_colors: int = 256
    ) -> Dict:
        """Create metadata for label image with image-label section."""
        metadata = {
            "multiscales": [{
                "version": self.config.ome_version,
                "name": name,
                "axes": axes,
                "datasets": datasets
            }]
        }

        # Generate colors if not provided
        if colors is None:
            from .metadata_utils import generate_default_label_colors
            colors = generate_default_label_colors(num_default_colors)

        # Add image-label section
        image_label = {
            'version': self.config.ome_version,
            'colors': colors
        }

        if source_image_path:
            image_label['source'] = {'image': source_image_path}

        metadata['image-label'] = image_label

        return metadata

    def create_root_metadata(
        self,
        image_multiscales: Optional[Dict] = None,
        has_labels: bool = False,
        image_name: str = 'image'
    ) -> Dict:
        """Create root .zattrs metadata."""
        metadata = {}

        if image_multiscales:
            # Adjust paths to include image_key prefix
            adjusted_datasets = []
            for ds in image_multiscales.get('datasets', []):
                new_ds = ds.copy()
                original_path = ds.get('path', '0')
                new_ds['path'] = f"{self.config.image_key}/{original_path}"
                adjusted_datasets.append(new_ds)

            metadata['multiscales'] = [{
                'version': self.config.ome_version,
                'name': image_name,
                'axes': image_multiscales.get('axes', []),
                'datasets': adjusted_datasets
            }]

        if has_labels:
            metadata['labels'] = [self.config.labels_container]

        return metadata

    # ========================================================================
    # Metadata Writing
    # ========================================================================

    def write_image_metadata(
        self,
        multiscales: Dict,
        name: str = 'image'
    ) -> None:
        """Write image metadata to image group .zattrs."""
        path = self.get_image_data_path()
        self._write_zgroup(path)
        metadata = self.create_image_multiscales_metadata(
            axes=multiscales.get('axes', []),
            datasets=multiscales.get('datasets', []),
            name=name
        )
        self._write_zattrs(path, metadata)

    def write_labels_container_metadata(
        self,
        label_names: Optional[List[str]] = None
    ) -> None:
        """Write labels container metadata, merging with existing labels."""
        path = self.get_labels_container_path()
        self._write_zgroup(path)

        # Read existing labels to merge (avoid clobbering previous label entries)
        existing_labels = []
        zattrs_path = os.path.join(path, '.zattrs')
        if os.path.exists(zattrs_path):
            try:
                with open(zattrs_path, 'r') as f:
                    existing_metadata = json.load(f)
                existing_labels = existing_metadata.get('labels', [])
            except (json.JSONDecodeError, IOError):
                pass

        # Merge: union of existing + new label names, preserving order
        new_labels = label_names or [self.config.label_name]
        merged = list(dict.fromkeys(existing_labels + new_labels))

        metadata = {"labels": merged}
        self._write_zattrs(path, metadata)

    def write_label_image_metadata(
        self,
        multiscales: Dict,
        name: str = 'segmentation',
        colors: Optional[List[Dict]] = None,
        source_image_path: Optional[str] = None,
        label_name: Optional[str] = None
    ) -> None:
        """Write label image metadata."""
        if source_image_path is None:
            source_image_path = f"../../{self.config.image_key}"

        path = self.get_label_data_path(label_name)
        self._write_zgroup(path)
        metadata = self.create_label_image_metadata(
            axes=multiscales.get('axes', []),
            datasets=multiscales.get('datasets', []),
            name=name,
            colors=colors,
            source_image_path=source_image_path
        )
        self._write_zattrs(path, metadata)

    def write_root_metadata(
        self,
        image_multiscales: Optional[Dict] = None,
        has_labels: bool = False,
        image_name: str = 'image',
        ome_xml: Optional[str] = None
    ) -> None:
        """Write root .zattrs metadata, merging with existing if present."""
        self._write_zgroup(self.output_path)

        # Read existing metadata if present (to preserve image multiscales when adding labels)
        existing_metadata = {}
        zattrs_path = os.path.join(self.output_path, '.zattrs')
        if os.path.exists(zattrs_path):
            try:
                with open(zattrs_path, 'r') as f:
                    existing_metadata = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_metadata = {}

        # Create new metadata
        metadata = self.create_root_metadata(
            image_multiscales=image_multiscales,
            has_labels=has_labels,
            image_name=image_name
        )

        # Merge: preserve existing multiscales if we're not providing new ones
        if image_multiscales is None and 'multiscales' in existing_metadata:
            metadata['multiscales'] = existing_metadata['multiscales']

        # Merge: preserve existing labels if present and we're adding more
        if 'labels' in existing_metadata and 'labels' in metadata:
            # Combine labels lists without duplicates
            existing_labels = set(existing_metadata.get('labels', []))
            new_labels = set(metadata.get('labels', []))
            metadata['labels'] = list(existing_labels | new_labels)
        elif 'labels' in existing_metadata and 'labels' not in metadata:
            metadata['labels'] = existing_metadata['labels']

        # Preserve or set ome_xml (consistent with non-nested Zarr2 mode)
        final_ome_xml = ome_xml or existing_metadata.get('ome_xml')
        if final_ome_xml:
            metadata['ome_xml'] = final_ome_xml

        self._write_zattrs(self.output_path, metadata)
