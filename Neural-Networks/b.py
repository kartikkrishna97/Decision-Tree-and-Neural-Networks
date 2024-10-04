import numpy as np 
import sys
import pdb
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import math
from tqdm import tqdm
from sklearn.metrics import classification_report


def get_data(x_path, y_path):
    '''
    Args:
        x_path: path to x file
        y_path: path to y file
    Returns:
        x: np array of [NUM_OF_SAMPLES x n]
        y: np array of [NUM_OF_SAMPLES]
    '''
    x = np.load(x_path)
    y = np.load(y_path)

    y = y.astype('float')
    x = x.astype('float')

    #normalize x:
    x = 2*(0.5 - x/255)
    return x, y





x_train_path = 'x_train.npy'
y_train_path = 'y_train.npy'

X_train, Y_train = get_data(x_train_path, y_train_path)
x_test_path = 'x_test.npy'
y_test_path = 'y_test.npy'

X_test, Y_test = get_data(x_test_path, y_test_path)


mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128, 64), 
                    activation='relu', 
                    solver='sgd', 
                    max_iter=250, 
                    random_state=42, 
                    batch_size=32,
                    alpha=0,
                    tol = 1e-2, 
                    learning_rate='invscaling')

mlp.fit(X_train, Y_train)
y_pred = mlp.predict(X_test)

print('for 1 layers')

print(classification_report(Y_test, y_pred))
