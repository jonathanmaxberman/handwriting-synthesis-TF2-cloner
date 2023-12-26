import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set the path to your directory containing the numpy files
directory = "Data/Strokes"

# Load and store sequences from numpy files
sequences = []
for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        file_path = os.path.join(directory, filename)
        sequence = np.load(file_path)
        sequences.append(sequence)

# Pad the sequences
padded_sequences = pad_sequences(sequences, padding='post')
