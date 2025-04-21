import tensorstore as ts
import tempfile
import numpy as np
import subprocess
import shutil
import os

os.umask(0o0002)

def test_n5_to_n5():
    temp_dir = create_n5()
    command = f"""python -m tensorswitch \
        --task n5_to_n5 \
        --base_path {temp_dir} \
        --output_path tests/temp/test_n5_to_n5"""
    subprocess.run(command.split())
    shutil.rmtree(temp_dir)
    return None

def test_n5_to_n5_submit():
    temp_dir = create_n5(os.getcwd() + "/submit_temp/test_n5_to_n5.n5")
    command = f"""python -m tensorswitch \
        --task n5_to_n5 \
        --base_path {temp_dir} \
        --output_path tests/temp/test_n5_to_n5
        --submit"""
    subprocess.run(command.split())
    shutil.rmtree(temp_dir)
    return None

def create_n5(temp_dir=None):
    """Creates a simple N5 dataset in a temporary directory."""
    if temp_dir is None:
        temp_dir = tempfile.TemporaryDirectory().name

    # ✅ Corrected chunk layout specification
    n5_spec = {
        'driver': 'n5',
        'kvstore': {'driver': 'file', 'path': temp_dir},
        'schema': {
            'rank': 3,
            'dtype': 'uint16',
            'domain': {'shape': [256, 256, 256]},
            'chunk_layout': {
                'chunk':{
                    'shape': [128, 128, 128]
                }
            }
        }
    }

    # ✅ Open and create the N5 dataset
    array = ts.open(n5_spec, create=True, open=True).result()
   
    # Populate the dataset with random data
    data = np.random.randint(0, 65535, size=(256, 256, 256), dtype=np.uint16)
    array[...] = data

    print(f"✅ N5 dataset created at: {temp_dir}")
    return temp_dir  # Return path to N5 dataset

if __name__ == "__main__":
    test_n5_to_n5()
    test_n5_to_n5_submit()
