import os
import re
from autotracecenter import tracetovector
def process_lines_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    processed_lines = {}
    for line in lines:
        if not line.startswith("#"):
            parts = line.split()
            if len(parts) > 1:
                key = parts[0]
                text = ' '.join(parts[6:]).replace('|', ' ')
                processed_lines[key] = text
    return processed_lines

def process_directory(root_dir, lines_data):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.png'):
                file_key = file[:-4]  # Remove '.png' from filename to match key
                if file_key in lines_data:
                    image_path = os.path.join(root, file)
                    input_string = lines_data[file_key]
                    tracetovector(image_path=image_path, input_string=input_string, path_to_npy='./data/tensors', name=f'{file_key}')

# Main execution
lines_data = process_lines_txt('./Data/lines.txt')
process_directory('./Data/lines', lines_data)
