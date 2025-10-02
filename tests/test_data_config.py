"""
Test data configuration for TensorSwitch tests.
Defines paths to real and synthetic test data.
"""

import os

# Base paths
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TESTS_DIR, 'test_data')
SYNTHETIC_DIR = os.path.join(TEST_DATA_DIR, 'synthetic')
REAL_DATA_DIR = os.path.join(TESTS_DIR, 'real_test_data')
OUTPUTS_DIR = os.path.join(TEST_DATA_DIR, 'outputs')

# Real test data - use middle chunks for faster testing
REAL_TEST_DATA = {
    'nd2': {
        'primary': {
            'path': os.path.join(REAL_DATA_DIR, 'nd2', '202507013_DID02_Brain2_4%PFA_Atto488_40XW_005.nd2'),
            'size': '5.7GB',
            'chunk_count': 10,  # Process 10 middle chunks
            'description': 'ND2 brain imaging data'
        },
        'secondary': {
            'path': os.path.join(REAL_DATA_DIR, 'nd2', '20250903_FlyID08-01_2ndGel_1%_0.8%_1hr_Atto488_40XW_006.nd2'),
            'size': '3.2GB',
            'chunk_count': 10,
            'description': 'ND2 fly imaging data'
        }
    },
    'ims': {
        'primary': {
            'path': os.path.join(REAL_DATA_DIR, 'ims', 'Pat16_MF6_2024-09-25_14.02.48_F0.ims'),
            'size': '3.8GB',
            'chunk_count': 10,
            'description': 'IMS microscopy data'
        }
    },
    'tif': {
        'primary': {
            'path': os.path.join(REAL_DATA_DIR, 'tif', '20250414_1p75-fold_8xbin_nuclear_segmentation.tif'),
            'size': '961MB',
            'chunk_count': 10,
            'description': 'TIFF nuclear segmentation data'
        }
    }
}

# Synthetic test data - small files for quick testing
SYNTHETIC_TEST_DATA = {
    'tif': os.path.join(SYNTHETIC_DIR, 'test_stack.tif'),
    'ims': os.path.join(SYNTHETIC_DIR, 'test_image.ims'),
    'nd2_like': os.path.join(SYNTHETIC_DIR, 'test_nd2_like.tif'),
    'n5': os.path.join(SYNTHETIC_DIR, 'test_volume.n5'),
    'zarr2': os.path.join(SYNTHETIC_DIR, 'test_zarr2.zarr'),
    'zarr3': os.path.join(SYNTHETIC_DIR, 'test_zarr3.zarr'),
}

# Output paths for test results
def get_output_path(test_name):
    """Get output path for a test."""
    return os.path.join(OUTPUTS_DIR, test_name)
