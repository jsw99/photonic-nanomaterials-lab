import numpy as np

def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy
    -> param y_true: list of true values
    -> param y_pred: list of predicted values
    -> return: accuracy score 
    """
    # Normalization, as we consider only the shape of a curve
    y_true_norm = (y_true-np.min(y_true)) / (np.max(y_true)-np.min(y_true))
    y_pred_norm = (y_pred-np.min(y_pred)) / (np.max(y_pred)-np.min(y_pred))
    # Intitializing variable to store count of correctly predicted classes
    correct_predictions = 0
    
    for yt, yp in zip(y_true_norm, y_pred_norm):
        if yt == yp: 
            correct_predictions += 1
    
    # Return accuracy
    return correct_predictions / y_true_norm.shape[0]