import tensorstore as ts
import tempfile
import numpy as np
import subprocess
import shutil
import os

os.umask(0o0002)

def test_n5_to_zarr2():
    temp_dir = create_n5_forZarr2()
    command = f"""python -m tensorswitch \
        --task n5_to_zarr2 \
        --base_path {temp_dir} \
        --output_path tests/temp/test_n5_to_zarr2 \
        --level 0 \
        --start_idx 0 \
        --memory_limit 50 \
        --num_volumes 1"""
    subprocess.run(command.split())
    shutil.rmtree(temp_dir)

def create_n5_forZarr2(temp_dir=None):
    """Creates a Zarr2-compatible N5 dataset with uint8 data."""
    if temp_dir is None:
        temp_dir = tempfile.TemporaryDirectory().name

    # ✅ Corrected chunk layout specification
    n5_spec = {
        'driver': 'n5',
        'kvstore': {'driver': 'file', 'path': temp_dir},
        'schema': {
            'rank': 3,
            'dtype': 'uint8',
            'domain': {'shape': [256, 256, 256]},
            'chunk_layout': {
                'chunk': {
                    'shape': [128, 128, 128]
                }
            }
        }
    }

    # ✅ Open and create the N5 dataset
    array = ts.open(n5_spec, create=True, open=True).result()
    
    # Populate the dataset with random data
    data = np.random.randint(0, 255, size=(256, 256, 256), dtype=np.uint8)
    array[...] = data

    print(f"✅ N5 dataset created at: {temp_dir}")
    return temp_dir

if __name__ == "__main__":
    test_n5_to_zarr2()