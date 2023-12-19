import numpy as np
import os

def load_text_data(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def create_char_to_int_mapping(text):
    unique_chars = sorted(list(set(text)))
    char_to_int = dict((c, i) for i, c in enumerate(unique_chars))
    return char_to_int

def encode_text(text, mapping):
    integer_encoded = [mapping[char] for char in text]
    return integer_encoded

# Example usage
data_directory = 'Data/chartest'  # Update with the path to your chars data
encoded_data = {}

for filename in os.listdir(data_directory):
    if filename.endswith('.npy'):
        file_path = os.path.join(data_directory, filename)
        text = np.load(file_path)
        char_to_int = create_char_to_int_mapping(text)
        encoded_data[filename] = encode_text(text, char_to_int)

# encoded_data now contains the integer-encoded text data
