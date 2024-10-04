from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import time 
from utils import  get_one_hot_array
from functools import reduce
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


train_path = "train.csv"
val_path = 'val.csv'
test_csv = 'test.csv'

X_train, Y_train, type, attributes = get_one_hot_array(train_path)
X_val, Y_val, type, attributes = get_one_hot_array(val_path)
X_test, Y_test, type, attributes = get_one_hot_array(val_path)

ccp_alpha_params = [0.001, 0.01, 0.1, 0.2]

tree = DecisionTreeClassifier(max_depth=45, ccp_alpha=ccp_alpha_params[0])
tree.fit(X_train, Y_train)


y_pred_train = tree.predict(X_train)
initial_accuracy_train = accuracy_score(Y_train, y_pred_train)

y_pred_test = tree.predict(X_test)
initial_accuracy_test = accuracy_score(Y_test, y_pred_test)

y_pred_val = tree.predict(X_val)
initial_accuracy_val = accuracy_score(Y_val, y_pred_val)

print(f"training accuracy after pruning is {initial_accuracy_train}")
print(f"validation accuracy after pruning is {initial_accuracy_val}")
print(f"test accuracy after pruning is {initial_accuracy_test}")





