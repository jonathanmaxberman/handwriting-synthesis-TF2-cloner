import os
import re
import shutil

def extract_identifier(filename):
    # Extract the unique identifier from the filename (e.g., '01-000u-00' from '01-000u-00styletest.npy')
    match = re.search(r'(\d+-\w+-\d+)', filename)
    return match.group(1) if match else None

def organize_files(src_directory, dst_directory, file_suffix, new_prefix):
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)

    file_map = {}
    files = [f for f in os.listdir(src_directory) if f.endswith(file_suffix)]
    for filename in files:
        identifier = extract_identifier(filename)
        if identifier:
            file_map[identifier] = filename

    for i, (identifier, filename) in enumerate(file_map.items(), start=1):
        new_filename = f"{new_prefix}_{i:04d}.npy"
        src_path = os.path.join(src_directory, filename)
        dst_path = os.path.join(dst_directory, new_filename)
        shutil.move(src_path, dst_path)
        print(f"Moved '{src_path}' to '{dst_path}'")

# Paths
data_directory = 'Data'
chartest_directory = 'Data/chartest'
styletest_directory = 'Data/styletest'

# Organize chartest and styletest files
organize_files(data_directory, chartest_directory, 'chartest.npy', 'char')
organize_files(data_directory, styletest_directory, 'styletest.npy', 'style')
