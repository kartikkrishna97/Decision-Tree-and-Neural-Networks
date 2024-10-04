import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
label_encoder = None 

train_path = 'train.csv'

def preprocess_data(path):

    X = pd.read_csv(path)
    X =  X.drop('Unnamed: 0', axis=1)
    need_label_encoding = ['team','host','opp','month', 'day_match']

    type = []   
    attributes = []
    for attribute in X.columns:
        if attribute in need_label_encoding:
            X[attribute] = pd.factorize(X[attribute])[0] + 1
            type.append(X.columns.get_loc(attribute))
        if attribute == 'result':
            continue
        else:
            attributes.append(attribute)
    
    Y = X['result']
    X = X.drop('result', axis=1)
    X = X.rename(columns=X.iloc[0]).drop(X.index[0])
    Z = Y.drop(Y.index[0])
    X = X.to_numpy()
    Z = Z.to_numpy()
    return X, Z, type, attributes

label_encoder = None 

def get_one_hot_array(file_name):
    global label_encoder
    data = pd.read_csv(file_name)
    
    need_label_encoding = ['team','host','opp','month', 'day_match']
    if(label_encoder is None):
        label_encoder = OneHotEncoder(sparse_output = False)
        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    type = np.arange(data_1.shape[1])
    
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    attributes = np.arange(X.shape[1])
    return X.to_numpy(), y.to_numpy(), attributes, type








 



    