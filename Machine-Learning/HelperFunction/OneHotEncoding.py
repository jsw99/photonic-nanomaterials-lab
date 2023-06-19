import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

nucleotide = {'A': 0, 'C': 1, 'G':2, 'T': 3}
input_string = {} # Record the name
input_matrix = {} # Record the matrix

# Convert a sequence in string form into matrix form
def string_to_psv(seq_string, i):
    conversion = np.array([nucleotide[j] for j in seq_string])
    encoded = to_categorical(conversion, dtype='int', num_classes=4)
    matrix = np.transpose(encoded).reshape(4,12)
    input_string[i] = seq_string
    input_matrix[i] = matrix
