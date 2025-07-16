import os
import sys
import numpy as np
import tifffile

def estimate_chunks_per_z(tiff_path, chunk_shape=(64, 64, 64)):
    # List .tif or .tiff files
    file_list = sorted([
        f for f in os.listdir(tiff_path)
        if f.lower().endswith((".tif", ".tiff"))
    ])
    if not file_list:
        raise ValueError(f"No TIFF files found in: {tiff_path}")
    
    z = len(file_list)
    sample = tifffile.imread(os.path.join(tiff_path, file_list[0]))
    y, x = sample.shape

    shape = np.array((z, y, x))
    chunk_shape = np.array(chunk_shape)
    chunk_counts = np.ceil(shape / chunk_shape).astype(int)

    chunks_per_z = chunk_counts[1] * chunk_counts[2]
    print(f"Volume shape: (z={z}, y={y}, x={x})")
    print(f"Chunks per z-slice: {chunks_per_z}")
    return chunks_per_z

def main():
    if len(sys.argv) != 3:
        print("Usage: python z_to_chunk_index.py <tiff_folder> <z_index>")
        sys.exit(1)

    tiff_path = sys.argv[1]
    z_index = int(sys.argv[2])

    chunks_per_z = estimate_chunks_per_z(tiff_path)
    start_chunk_index = z_index * chunks_per_z

    print(f"Z-slice {z_index} → Start chunk index: {start_chunk_index}")

if __name__ == "__main__":
    main()
