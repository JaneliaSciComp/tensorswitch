import tensorstore as ts
import tempfile
import numpy as np
import subprocess
import shutil
import os

os.umask(0o0002)

def test_zarr3_to_downsample_noshard_zarr3():
    temp_dir = create_zarr3_dataset()
    output_dir = "tests/temp/test_zarr3_to_downsample_noshard_zarr3"

    command = f"""python -m tensorswitch \
        --task downsample_shard_zarr3 \
        --base_path {temp_dir} \
        --output_path {output_dir} \
        --level 1 \
        --start_idx 0 \
        --use_shard 0"""

    subprocess.run(command.split())
    shutil.rmtree(temp_dir)

def create_zarr3_dataset(temp_dir=None):
    if temp_dir is None:
        temp_dir = tempfile.TemporaryDirectory().name

    spec = {
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': f"{temp_dir}/multiscale/s0"},
        'metadata': {
            'shape': [64, 64, 64],
            'chunk_grid': {
                'name': 'regular',
                'configuration': {'chunk_shape': [64, 64, 64]}
            },
            'chunk_key_encoding': {'name': 'default'},
            'data_type': 'uint8',
            'node_type': 'array',
            'codecs': [{'name': 'zstd', 'configuration': {'level': 5}}]
        }
    }

    arr = ts.open(spec, create=True, open=True).result()
    arr[...] = np.random.randint(0, 255, size=(64, 64, 64), dtype=np.uint8)
    print(f"✅ Zarr3 dataset (no shard) created at: {temp_dir}")
    return temp_dir

if __name__ == "__main__":
    test_zarr3_to_downsample_noshard_zarr3()
