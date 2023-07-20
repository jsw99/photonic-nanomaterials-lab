import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
import seaborn as sns

# Multiclass classification
def create_model_for_class(num_dense_layers, num_dense_nodes):
    
    model = Sequential()
    
    # Add an input layer with input shape being 4*12
    model.add(InputLayer(input_shape=(4, 12)))
    
    # Add a flatten layer
    model.add(Flatten())
    
    # The number of layers, number of neurons, activation function
    for i in range(num_dense_layers):
        model.add(Dense(units=num_dense_nodes, activation='relu'))
    
    # Last dense layer for classification over the whole spectrum, 640 wavelengths
    model.add(Dense(640*20))
    
    model.add(tf.keras.layers.Reshape((640, 20)))
    
    # Use Adam as optimizer
    optimizer = Adam(learning_rate=1e-3)
    
    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model

# Binary classification
def create_model_for_change(num_dense_layers, num_dense_nodes):
    
    model = Sequential()
    
    # Add an input layer with input shape being 4*12
    model.add(InputLayer(input_shape=(4, 12)))
    
    # Add a flatten layer
    model.add(Flatten())
    
    # The number of layers, number of neurons, activation function
    for i in range(num_dense_layers):
        model.add(Dense(units=num_dense_nodes, activation='relu'))
    
    # Last dense layer for regression over the whole spectrum, 640 wavelengths
    model.add(Dense(640, activation='sigmoid'))
    
    # Use Adam as optimizer
    optimizer = Adam(learning_rate=1e-3)
    
    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Connect logits layer to softmax layer
def create_probability_model_for_class(model):
    model_probability = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    return model_probability

# Define the predict function
@tf.function(reduce_retracing=True)
def predict(model, x):
    return model(x)

# Convert change probability into class
def convert_change_probability_into_class(change_prob, threshold):
    change_temp = np.zeros((640,))
    for i in range(640):
        if change_prob[0][i] > threshold:
            change_temp[i] = 1
        else:
            change_temp[i] = 0

    return change_temp

# Plot class figures
def plot_class(class_prob, change_temp, dna_name, color):
    wavelength = np.loadtxt('/Users/jasonwang/photonic-nanomaterials-lab/Spectra/930~1367_640_wavelength.txt', usecols=0)

    fig = plt.figure(figsize=(13, (13-1.5)/1.618))
    ax = fig.add_axes([0.26, 0.15, 0.735, 0.735*13/(13-1.5)])

    ax.set_xlim(930, 1370)
    ax.set_ylim(-2.1, 2.1)
    ax.set_yticks(np.arange(-2,2.1,0.5))

    class_labels = np.arange(0.1, 2.1, 0.1)
    # Convert highest probabilities into corresponding classes
    class_20 = np.array([class_labels[i] for i in np.argmax(class_prob[0], axis=1)])

    class_40 = np.zeros((640,))
    for i in range(0, 640):
        if change_temp[i] == 0:
            class_40[i] = -class_20[i]
        else:
            class_40[i] = class_20[i]

    mp = ax.plot(wavelength, class_40, color=color, linewidth=1.5, label='{}-HiPco (Predicted)'.format(dna_name))

    ax.set_xlabel('Wavelength (nm)', fontsize=25, labelpad=20)
    ax.set_ylabel('Class', fontsize=25, labelpad=18)

    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15, top='on', direction='in', pad=15)
    ax.xaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6, top='on', direction='in')

    ax.yaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15/2.5, right='on', direction='in', pad=15)
    #ax.yaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6, right='on', direction='in')
    ax.tick_params(axis='y', which='minor', left=False)

    ax.legend(loc='best', fontsize=15, fancybox=True, framealpha=0.5)

    for i in ['right', 'left', 'top', 'bottom']:
        ax.spines[i].set_linewidth(2.5)

    plt.show()

# Set class_40[?] = 0, for comparing different curves' shapes
def set_mid_to_zero(class_40, i):
    class_40_norm = class_40 - (class_40[i]-0)
    return class_40_norm

# Convert 20 classes to 40 classes
def convert_20_to_40(class_20, change_temp):
    class_40 = np.zeros((640,))
    for i in range(0, 640):
        if change_temp[i] == 0:
            class_40[i] = -class_20[i]
        else:
            class_40[i] = class_20[i]

# Plot confusion matrix
def plot_confusion_matrix(prediction_class, test_class):
    pred = (prediction_class*10).astype(int)
    true = (test_class*10).astype(int)

    classes = [-2., -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1,
                -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
               0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,
               1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.]

    cfm = metrics.confusion_matrix(true, pred, labels=np.array(classes)*10)

    fig, ax = plt.subplots(figsize=(15*1.5, 6*1.5))
    sns.heatmap(cfm,
                fmt='.0f', cmap='coolwarm',
                square=True, annot=True, annot_kws={'size':'small', 'alpha':0.6}, linewidths=0.5,
                xticklabels=classes, yticklabels=classes,
                cbar_kws={"shrink": 1})

    #ax.xaxis.tick_top()
    ax.xaxis.set_tick_params(which='major', labelsize=8, width=0.5, length=2.5, direction='out', pad=3)
    ax.yaxis.set_tick_params(which='major', labelsize=8, width=0.5, length=2.5, direction='out', pad=3)
    ax.set_xlabel('Predicted Class', fontsize=15, labelpad=10)
    ax.set_ylabel('True Class', fontsize=15, labelpad=10)

    plt.show()

    acc = np.sum(np.diag(cfm))\
            + np.sum(np.diag(cfm, k=1)) + np.sum(np.diag(cfm, k=-1))\
            + np.sum(np.diag(cfm, k=2)) + np.sum(np.diag(cfm, k=-2))

    return acc