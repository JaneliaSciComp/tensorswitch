import tensorstore as ts
import os
import time
import psutil
from ..utils import (get_chunk_domains, n5_store_spec, zarr2_store_spec, create_output_store,
                     commit_tasks, print_processing_info, get_total_chunks_from_store,
                     get_tensorstore_context, detect_source_order, create_zarr2_ome_metadata,
                     write_zarr2_group_metadata, read_n5_attributes, extract_n5_metadata)

def convert(base_path, output_path, level, start_idx=0, stop_idx=None, memory_limit=50, **kwargs):
    """Convert N5 to Zarr2 format."""
    #n5_level_path = os.path.join(base_path, f"s{level}")
    #zarr_level_path = os.path.join(output_path, f"s{level}")
    n5_level_path = f"{base_path}"
    zarr_level_path = f"{output_path}"
    os.makedirs(zarr_level_path, exist_ok=True)

    # Add TensorStore context to limit concurrency to LSF allocation
    n5_spec = n5_store_spec(n5_level_path)
    n5_spec['context'] = get_tensorstore_context()
    n5_store = ts.open(n5_spec).result()
    shape, chunks = n5_store.shape, n5_store.chunk_layout.read_chunk.shape

    # Detect source data order
    source_order_info = detect_source_order(n5_store)
    print(f"Source data order: {source_order_info['description']}")
    print(f"  Inner order: {source_order_info['inner_order']}")
    print(f"  Detected axes: {source_order_info['suggested_axes']}")

    # Determine output order: preserve source order unless user overrides
    if 'use_fortran_order' in kwargs:
        # User explicitly set use_fortran_order → respect it
        use_fortran_order = kwargs.get('use_fortran_order', False)
        if use_fortran_order:
            print(f"✓ Using F-order output (user override)")
        else:
            print(f"✓ Using C-order output (user override)")
    else:
        # No user override → preserve source order (auto-detect)
        use_fortran_order = source_order_info['is_fortran_order']
        if use_fortran_order:
            print(f"✓ Preserving F-order from source (auto-detected)")
        else:
            print(f"✓ Preserving C-order from source (auto-detected)")

    zarr2_spec = zarr2_store_spec(zarr_level_path, shape, chunks, use_fortran_order=use_fortran_order)
    zarr2_spec['context'] = get_tensorstore_context()
    zarr2_store = create_output_store(zarr2_spec)

    total_chunks = get_total_chunks_from_store(zarr2_store, chunk_shape=chunks)
    print(f" Total chunks to write: {total_chunks}")
    print(f" Writing from chunk {start_idx} to {stop_idx}")


    if stop_idx is None:
        stop_idx = total_chunks

    print_processing_info(level, start_idx, stop_idx, total_chunks)

    # Process chunks with transaction-per-chunk pattern (Mark's fix)
    tasks = []
    linear_indices_to_process = range(start_idx, stop_idx)
    for chunk_domain in get_chunk_domains(chunks, zarr2_store, linear_indices_to_process=linear_indices_to_process):
        # Create transaction per chunk to prevent loading all data simultaneously
        with ts.Transaction() as txn:
            task = zarr2_store[chunk_domain].with_transaction(txn).write(n5_store[chunk_domain])
        tasks.append(task)

    # Wait for all tasks to complete
    for task in tasks:
        task.result()

    print(f"Conversion complete for {n5_level_path} to {zarr_level_path}")

    # Create OME metadata with correct axes if this is level 0
    if level == 0:
        try:
            print("\n" + "="*80)
            print("CREATING OME-NGFF METADATA FOR ZARR2")
            print("="*80)

            # Read N5 attributes from both level and root directories
            source_attrs, root_attrs = read_n5_attributes(base_path)

            # Extract metadata from N5 (prefer root_attrs for multiscales metadata)
            attrs_to_check = root_attrs if root_attrs else source_attrs
            n5_metadata = extract_n5_metadata(attrs_to_check, level=level)

            # Get axes order: first try from N5 metadata, then fall back to detected order
            axes_order = n5_metadata['axes_order']
            if not axes_order:
                axes_order = source_order_info['suggested_axes']
                print(f"No axes in N5 metadata → using detected axes: {axes_order}")
            else:
                print(f"Axes order (from N5 metadata): {axes_order}")

            # Create pixel_sizes dict if we have voxel sizes
            pixel_sizes = None
            if n5_metadata['voxel_sizes_um']:
                # Map voxel sizes to dict based on axes_order
                pixel_sizes = {
                    axes_order[i]: n5_metadata['voxel_sizes_um'][i]
                    for i in range(len(axes_order))
                }
                print(f"Voxel sizes: {n5_metadata['voxel_sizes_um']} µm → {pixel_sizes}")

            # Create OME metadata
            zarr2_metadata = create_zarr2_ome_metadata(
                ome_xml=None,  # N5 doesn't have OME-XML
                array_shape=shape,
                image_name=n5_metadata['dataset_name'] or os.path.basename(output_path),
                pixel_sizes=pixel_sizes,
                axes_order=axes_order  # Pass axes order
            )

            # Write root .zattrs
            write_zarr2_group_metadata(output_path, zarr2_metadata)
            print("✓ Root .zattrs created with OME-NGFF metadata")
            print(f"✓ Axes order in metadata: {axes_order}")

        except Exception as e:
            print(f"Warning: Could not create root .zattrs: {e}")
            print("  Conversion successful, but root metadata is missing")