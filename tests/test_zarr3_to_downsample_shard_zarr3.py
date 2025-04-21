import tensorstore as ts
import tempfile
import numpy as np
import subprocess
import shutil
import os

os.umask(0o0002)

def test_zarr3_to_downsample_shard_zarr3():
    temp_dir = create_zarr3()
    command = f"""python -m tensorswitch \
        --task downsample_shard_zarr3 \
        --base_path {temp_dir} \
        --output_path tests/temp/test_zarr3_to_downsample_shard_zarr3 \
        --level 1 \
        --start_idx 0 \
        --memory_limit 50 \
        --use_shard 1 \
        --num_volumes 1"""
    subprocess.run(command.split())
    shutil.rmtree(temp_dir)

def create_zarr3(temp_dir=None):
    if temp_dir is None:
        temp_dir = tempfile.TemporaryDirectory().name

    zarr3_spec = {
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': f"{temp_dir}/multiscale/s0"},
        'metadata': {
            'shape': [64, 64, 64],
            'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': [32, 32, 32]}},
            'chunk_key_encoding': {'name': 'default'},
            'data_type': 'uint8',
            'node_type': 'array',
            'codecs': [
                {'name': 'zstd', 'configuration': {'level': 5}}
            ]
        }
    }

    array = ts.open(zarr3_spec, create=True).result()

    # Fill s0 with some data
    data = np.random.randint(0, 255, size=(64, 64, 64), dtype=np.uint8)
    array[...] = data

    print(f"✅ Zarr3 dataset created at: {temp_dir}")
    return temp_dir

if __name__ == "__main__":
    test_zarr3_to_downsample_shard_zarr3()

