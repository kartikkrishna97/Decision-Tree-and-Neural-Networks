import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
label_encoder = None 
import pandas as pd


def get_one_hot_array(file_name):
    global label_encoder
    data = pd.read_csv(file_name)
    
    need_label_encoding = ['team','host','opp','month', 'day_match']  
    if(label_encoder is None):
        label_encoder = OneHotEncoder(sparse_output = False)
        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), 
                          columns = label_encoder.get_feature_names_out())
    dont_need_label_encoding = ["year","toss","bat_first","format","fow","score","rpo","result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    
    type = np.array([]) 
    
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    attributes = np.arange(X.shape[1])
    return X.to_numpy(), y.to_numpy(), attributes, type

def preprocess_data(path):

    X = pd.read_csv(path)
    # X =  X.drop('Unnamed: 0', axis=1)
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

import numpy as np



def entropy(labels):
    """
    Calculate entropy of a set of labels.
    
    H(S) = -Σ p(c) * log2(p(c))
    
    Args:
        labels: Array of binary labels (0 or 1)
    
    Returns:
        Entropy value
    """
    if len(labels) == 0:
        return 0
    
    n_positive = np.sum(labels == 1)
    n_negative = np.sum(labels == 0)
    total = len(labels)
    
    if n_positive == 0 or n_negative == 0:
        return 0
    
    p_pos = n_positive / total
    p_neg = n_negative / total
    

    return -p_pos * np.log2(p_pos) - p_neg * np.log2(p_neg)


def information_gain_categorical(features, labels, feature_idx):
    """
    Calculate information gain for a CATEGORICAL feature.
    
    Split data by unique feature values, calculate weighted entropy.
    
    Args:
        features: Data array
        labels: Target labels
        feature_idx: Index of feature to evaluate
    
    Returns:
        Information gain value
    """
    
    parent_entropy = entropy(labels)
    
    unique_values = np.unique(features[:, feature_idx])
    
    weighted_child_entropy = 0
    total_samples = len(labels)
    
    for value in unique_values:
        mask = features[:, feature_idx] == value
        subset_labels = labels[mask]
        
        weight = len(subset_labels) / total_samples
        weighted_child_entropy += weight * entropy(subset_labels)
    
    return parent_entropy - weighted_child_entropy


def information_gain_continuous(features, labels, feature_idx, split_value):
    """
    Calculate information gain for a CONTINUOUS feature at a given split point.
    
    Split data into left (<=) and right (>) based on split_value.
    
    Args:
        features: Data array
        labels: Target labels
        feature_idx: Index of feature to evaluate
        split_value: Threshold to split on
    
    Returns:
        Information gain value
    """
    parent_entropy = entropy(labels)
    
    left_mask = features[:, feature_idx] <= split_value
    right_mask = features[:, feature_idx] > split_value
    
    left_labels = labels[left_mask]
    right_labels = labels[right_mask]
    
    if len(left_labels) == 0 or len(right_labels) == 0:
        return 0
    
    total_samples = len(labels)
    left_weight = len(left_labels) / total_samples
    right_weight = len(right_labels) / total_samples
    
    weighted_child_entropy = (left_weight * entropy(left_labels) + 
                              right_weight * entropy(right_labels))
    
    return parent_entropy - weighted_child_entropy


def best_split_continuous(features, labels, feature_idx):
    """
    Find the best split point for a continuous feature.
    
    Try multiple candidate split points and return the best one.
    
    Args:
        features: Data array
        labels: Target labels
        feature_idx: Index of feature to evaluate
    
    Returns:
        best_split_value, best_information_gain
    """
    feature_values = features[:, feature_idx]
    

    unique_values = np.unique(feature_values)
    

    if len(unique_values) > 10:
        percentiles = np.linspace(10, 90, 9)
        candidate_splits = np.percentile(feature_values, percentiles)
    else:
        candidate_splits = (unique_values[:-1] + unique_values[1:]) / 2
    
    best_gain = -np.inf
    best_split = None
    
    for split_value in candidate_splits:
        gain = information_gain_continuous(features, labels, feature_idx, split_value)
        
        if gain > best_gain:
            best_gain = gain
            best_split = split_value
    
    return best_split, best_gain


def MutualInformation(features, targets, feature_index, type_indices):
    """
    SIMPLIFIED VERSION - replaces your complex implementation.
    
    Calculate information gain for a feature (categorical or continuous).
    
    Args:
        features: Data array
        targets: Labels
        feature_index: Which feature to evaluate
        type_indices: Set of indices for categorical features
    
    Returns:
        Information gain value
    """
    if feature_index in type_indices:
        return information_gain_categorical(features, targets, feature_index)
    else:
        _, best_gain = best_split_continuous(features, targets, feature_index)
        return best_gain


def best_attribute(features, labels, type_indices):
    """
    Find feature with maximum information gain.
    
    Args:
        features: Data array
        labels: Target labels
        type_indices: Set of categorical feature indices
    
    Returns:
        best_feature_idx, best_cutoff_value
    """
    best_feature = -1
    best_cutoff = None
    max_gain = -np.inf
    
    num_features = features.shape[1]
    
    for idx in range(num_features):
        if idx in type_indices:
            gain = information_gain_categorical(features, labels, idx)
            cutoff = None
        else:
            cutoff, gain = best_split_continuous(features, labels, idx)
        
        if gain > max_gain:
            max_gain = gain
            best_feature = idx
            best_cutoff = cutoff
    
    return best_feature, best_cutoff











 



    
