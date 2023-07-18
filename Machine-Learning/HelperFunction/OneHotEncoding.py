import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pandas as pd

nucleotide: dict = {'A': 0, 'C': 1, 'G':2, 'T': 3}
input_string: dict = {} # Record the name
input_matrix: dict = {} # Record the matrix

# Convert a sequence in string form into matrix form
def string_to_psv(seq_string, i):
    conversion = np.array([nucleotide[j] for j in seq_string])
    encoded = to_categorical(conversion, dtype='int', num_classes=4)
    matrix = np.transpose(encoded).reshape(4,12)
    input_string[i] = seq_string
    # Use the wrapper around the numpy array to store numpy array into pandas cell
    input_matrix[i] = matrix

# Available data
string_to_psv('TTTTCCCCCCCT', 13)
string_to_psv('ATTTTTTTTTTA', 18)
string_to_psv('TCTCTTGGACCC', 23)
string_to_psv('GGGGTTTTGGGG', 24)
string_to_psv('GCAGCGTGACTT', 26)
string_to_psv('AACACGGCCCTC', 27)
string_to_psv('AGCACAACACGG', 33)
string_to_psv('ACACACCATCAG', 35)
string_to_psv('AGCAGCACACGA', 36)
string_to_psv('AGCACCAGACAG', 40)
string_to_psv('ACCACGATCCTC', 41)
string_to_psv('ACGCACCGACAG', 42)

#df = pd.DataFrame(input_matrix)

#df.to_csv('/Users///ML/training_data/X_train/X_train.csv', mode='a', header=True, index=True)

np.savez(
    '/Users///ML/training_data/X_train/X_train.npz', 
    x13 = input_matrix[13],
    x18 = input_matrix[18],
    x23 = input_matrix[23],
    x24 = input_matrix[24],
    x26 = input_matrix[26],
    x27 = input_matrix[27],
    x33 = input_matrix[33],
    x35 = input_matrix[35],
    x36 = input_matrix[36],
    x40 = input_matrix[40],
    x41 = input_matrix[41],
    x42 = input_matrix[42],
    )