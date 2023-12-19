import os
import numpy as np
import re

def correct_text(data):
    # Ensure the data is a string before applying regex
    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            # Handle 0-dimensional numpy array
            text = data.item().decode('utf-8') if isinstance(data.item(), bytes) else data.item()
        elif data.size > 0:
            # Assuming the first element of the array is the target text
            text_element = data[0]
            text = text_element.decode('utf-8') if isinstance(text_element, bytes) else text_element
        else:
            # Empty array case
            text = ''
    elif isinstance(data, bytes):
        # Directly decode bytes to string
        text = data.decode('utf-8')
    else:
        # If it's already a string or other types
        text = data

    # Use regex to remove leading numbers (coordinates) but keep the rest of the text
    corrected_text = re.sub(r'^[\d\s]+', '', text)
    return corrected_text

def process_npy_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            data = np.load(file_path, allow_pickle=True)
            corrected_text = correct_text(data)
            corrected_text = corrected_text.encode('utf-8')
            uint8_array = np.frombuffer(corrected_text, dtype=np.uint8)
            print(corrected_text)
            print(file_path)
            # Save the corrected text back to the file
            np.save(file_path, uint8_array)

# Process all .npy files in the specified directory
process_npy_files('Data/chartest/')
