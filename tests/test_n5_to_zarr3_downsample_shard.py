import tensorstore as ts
import tempfile
import numpy as np
import subprocess
import shutil
import os

os.umask(0o0002)

def test_n5_to_zarr3_downsample_shard():
    temp_n5 = create_n5_dataset()
    zarr3_path = "tests/temp/test_n5_to_zarr3_downsample_shard/zarr3"
    final_path = "tests/temp/test_n5_to_zarr3_downsample_shard/final"

    # Step 1: Convert N5 to Zarr3 directly (writing zarr.json)
    convert_n5_to_zarr3(temp_n5, zarr3_path)

    # Step 2: Run downsampling + shard
    command = f"""python -m tensorswitch \
        --task downsample_shard_zarr3 \
        --base_path {zarr3_path} \
        --output_path {final_path} \
        --level 1 \
        --use_shard 1"""
    
    subprocess.run(command.split())
    shutil.rmtree(temp_n5)

def create_n5_dataset():
    """Create small uint8 N5 dataset"""
    temp_dir = tempfile.TemporaryDirectory().name
    spec = {
        'driver': 'n5',
        'kvstore': {'driver': 'file', 'path': temp_dir},
        'schema': {
            'rank': 3,
            'dtype': 'uint8',
            'domain': {'shape': [64, 64, 64]},
            'chunk_layout': {'chunk': {'shape': [32, 32, 32]}}
        }
    }
    arr = ts.open(spec, create=True, open=True).result()
    arr[...] = np.random.randint(0, 255, size=(64, 64, 64), dtype=np.uint8)
    print(f"✅ N5 dataset created at: {temp_dir}")
    return temp_dir

def convert_n5_to_zarr3(n5_path, zarr3_path):
    """Convert small N5 to Zarr3 layout expected by downsampling step"""
    n5_store = ts.open({
        'driver': 'n5',
        'kvstore': {'driver': 'file', 'path': n5_path}
    }).result()

    zarr_spec = {
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': f"{zarr3_path}/multiscale/s0"},
        'metadata': {
            'shape': n5_store.shape,
            'chunk_grid': {
                'name': 'regular',
                'configuration': {'chunk_shape': list(n5_store.chunk_layout.read_chunk.shape)}
            },
            'chunk_key_encoding': {'name': 'default'},
            'data_type': n5_store.dtype.name,
            'node_type': 'array',
            'codecs': [
                {'name': 'bytes', 'configuration': {'endian': 'little'}},
                {'name': 'zstd', 'configuration': {'level': 5, 'checksum': False}}
            ]
        }
    }

    zarr_store = ts.open(zarr_spec, create=True, open=True).result()
    zarr_store[...] = n5_store[...]
    print(f" Converted to Zarr3 at: {zarr3_path}/multiscale/s0")

if __name__ == "__main__":
    test_n5_to_zarr3_downsample_shard()

