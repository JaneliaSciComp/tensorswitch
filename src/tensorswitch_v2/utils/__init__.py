# Copied from tensorswitch/utils.py for v2 independence
"""
Utility functions for tensorswitch_v2.

This package provides independent utility functions for tensorswitch_v2,
eliminating dependencies on tensorswitch (v1).

Modules:
- tensorstore_utils: TensorStore specs and context
- chunk_utils: Chunk/shard operations
- metadata_utils: OME-NGFF metadata functions
- pyramid_utils: Pyramid planning functions
- format_loaders: TIFF, ND2, IMS, CZI loaders
- ome_structure: OME-NGFF nested structure management
- folder_discovery: Auto-detection and classification of datasets in folders
"""

from .tensorstore_utils import (
    get_dtype_name,
    get_tensorstore_context,
    get_kvstore_spec,
    get_input_driver,
    get_zarr_store_spec,
    zarr3_store_spec,
    zarr2_store_spec,
    downsample_spec,
    detect_source_order,
    n5_store_spec,
    adaptive_spatial_chunk,
    build_default_shape,
)

from .chunk_utils import (
    get_chunk_domains,
    get_total_chunks_from_store,
    get_chunk_linear_indices_in_shard,
)

from .metadata_utils import (
    update_ome_metadata_if_needed,
    update_ome_multiscale_metadata,
    update_ome_multiscale_metadata_zarr2,
    auto_detect_max_level,
    write_zarr3_group_metadata,
    create_zarr3_ome_metadata,
    precreate_zarr3_output,
    precreate_shard_directories,
    precreate_zarr3_metadata_safely,
    extract_omero_channels,
    generate_default_label_colors,
)

from .pyramid_utils import (
    calculate_pyramid_plan,
    calculate_anisotropic_downsample_factors,
    calculate_num_multiscale_levels,
)

from .format_loaders import (
    load_tiff_stack,
    extract_tiff_ome_metadata,
    is_tiff_zstack_directory,
    load_nd2_stack,
    extract_nd2_ome_metadata,
    load_ims_stack,
    extract_ims_metadata,
    load_czi_stack,
    extract_czi_metadata,
)

from .ome_structure import (
    OMEStructure,
    OMEStructureConfig,
    OMEStructureZarr2,
    OMEStructureZarr2Config,
)

from .folder_discovery import (
    discover_datasets,
    is_neuroglancer_precomputed,
    is_zarr_dataset,
    is_n5_dataset,
    classify_dataset,
    classify_dataset_generic,
    validate_discovery_for_conversion,
    DiscoveredDataset,
    DiscoveryResult,
    SEGMENTATION_KEYWORDS,
)

from .resource_utils import (
    calculate_memory,
    calculate_wall_time,
    estimate_shard_info,
    calculate_job_resources,
)

__all__ = [
    # TensorStore utilities
    'get_tensorstore_context',
    'get_kvstore_spec',
    'get_input_driver',
    'get_zarr_store_spec',
    'zarr3_store_spec',
    'zarr2_store_spec',
    'downsample_spec',
    'detect_source_order',
    'n5_store_spec',
    'adaptive_spatial_chunk',
    'build_default_shape',
    # Chunk utilities
    'get_chunk_domains',
    'get_total_chunks_from_store',
    'get_chunk_linear_indices_in_shard',
    # Metadata utilities
    'update_ome_metadata_if_needed',
    'update_ome_multiscale_metadata',
    'update_ome_multiscale_metadata_zarr2',
    'auto_detect_max_level',
    'write_zarr3_group_metadata',
    'create_zarr3_ome_metadata',
    'precreate_zarr3_output',
    'precreate_shard_directories',
    'precreate_zarr3_metadata_safely',
    'extract_omero_channels',
    'generate_default_label_colors',
    # Pyramid utilities
    'calculate_pyramid_plan',
    'calculate_anisotropic_downsample_factors',
    'calculate_num_multiscale_levels',
    # Format loaders
    'load_tiff_stack',
    'extract_tiff_ome_metadata',
    'is_tiff_zstack_directory',
    'load_nd2_stack',
    'extract_nd2_ome_metadata',
    'load_ims_stack',
    'extract_ims_metadata',
    'load_czi_stack',
    'extract_czi_metadata',
    # OME-NGFF structure
    'OMEStructure',
    'OMEStructureConfig',
    'OMEStructureZarr2',
    'OMEStructureZarr2Config',
    # Folder discovery
    'discover_datasets',
    'is_neuroglancer_precomputed',
    'classify_dataset',
    'validate_discovery_for_conversion',
    'DiscoveredDataset',
    'DiscoveryResult',
    # Resource utilities
    'calculate_memory',
    'calculate_wall_time',
    'estimate_shard_info',
    'calculate_job_resources',
]
