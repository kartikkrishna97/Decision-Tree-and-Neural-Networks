import numpy as np

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time 
from functools import reduce
from utils import get_one_hot_array



X_train, Y_train, attributes, type = get_one_hot_array('train.csv')
X_test, Y_test, attributes, type = get_one_hot_array('test.csv')
X_val, Y_val, attributes, type = get_one_hot_array('val.csv')


def MutualInformation(features, targets, feature_index):

    num_positive = (targets == 1).sum()
    num_negative = (targets == 0).sum()
    if num_positive == 0 or num_negative == 0:
        entropy = 0
    else:
        prob_pos = num_positive / (num_positive + num_negative)
        prob_neg = num_negative / (num_positive + num_negative)
        entropy = -prob_pos * np.log(prob_pos) - prob_neg * np.log(prob_neg)

    mutual_info = []
    if feature_index in type:
        unique_features = np.unique(features[:, feature_index])
        for feature_val in unique_features:
            prob_feature = sum(features[:, feature_index] == feature_val) / len(features)
            if prob_feature == 0:
                continue
            targets_given_feature =  [targets[i] for i in range(len(features)) if features[i][feature_index] == feature_val]
            prob_target1_given_feature = sum(targets_given_feature) / len(targets_given_feature)
            prob_target0_given_feature = 1 - prob_target1_given_feature
            if prob_target1_given_feature == 0 or prob_target0_given_feature == 0:
                continue
            mutual_info.append(prob_feature * (prob_target0_given_feature * np.log(prob_target0_given_feature) + 
                                               prob_target1_given_feature * np.log(prob_target1_given_feature)))
    else:
        median_val = np.median(features[:, feature_index], axis=0)
        targets_left_median =  [targets[i] for i in range(len(features)) if features[i][feature_index] <= median_val]
        targets_right_median = [targets[i] for i in range(len(features)) if features[i][feature_index] > median_val]
        prob_target1_given_less_median = sum(targets_left_median) / len(targets_left_median)
        prob_target0_given_less_median = 1 - prob_target1_given_less_median
        if prob_target1_given_less_median != 0 and prob_target1_given_less_median != 1:
            mutual_info.append(len(targets_left_median)/len(targets) * 
                               (prob_target1_given_less_median * np.log(prob_target1_given_less_median) + 
                                prob_target0_given_less_median * np.log(prob_target0_given_less_median)))
        if len(targets_right_median) > 0:
            prob_target1_given_more_median = sum(targets_right_median) / len(targets_right_median)
            prob_target0_given_more_median = 1 - prob_target1_given_more_median
            if prob_target1_given_more_median != 0 and prob_target1_given_more_median != 1:
                mutual_info.append(len(targets_right_median) / len(targets) * 
                                   (prob_target1_given_more_median * np.log(prob_target1_given_more_median) + 
                                    prob_target0_given_more_median * np.log(prob_target0_given_more_median)))
    
    entropy_given_feature = reduce(lambda x, y: x - y, mutual_info, 0)
    return entropy - entropy_given_feature

def best_attribute(features, labels):
    best_attr, max_MI = -1, -np.inf
    num_of_attributes = len(features[0])
    
    for idx in range(num_of_attributes):
        mutual_info = MutualInformation(features, labels, idx)
        
        if mutual_info > max_MI:
            best_attr = idx
            max_MI = mutual_info
    
    cut_off = np.median(features[:, best_attr]) if best_attr not in type else None
    
    return best_attr, cut_off

class NodeStructure:
    def __init__(self, dataPoints=None, labels=None, treeDepth=0, depthLimit=3):
        self.childNodes = dict()
        self.feature = None
        self.cutoff = None
        self.isDiscrete = False
        self.isLeafNode = False
        self.result = None
        self.is_root = True

        if dataPoints is not None and labels is not None:
             self.createChildNodes(dataPoints, labels, treeDepth, depthLimit)

    def createChildNodes(self, dataPoints, labels, treeDepth, depthLimit):
        positiveLabelCount = sum(labels)
        negativeLabelCount = len(labels) - positiveLabelCount

        if positiveLabelCount == 0 or negativeLabelCount == 0 or treeDepth == depthLimit:
            self.isLeafNode = True
            self.result = 1 if positiveLabelCount > negativeLabelCount else 0
            return

        bestFeat, cutoffValue = best_attribute(dataPoints, labels)
        self.feature = bestFeat
        self.cutoff = cutoffValue
        self.result = 1 if positiveLabelCount > negativeLabelCount else 0

        if bestFeat in type: 
            uniqueValues = np.unique(dataPoints[:,bestFeat])
            for value in uniqueValues:
                dpSubset = dataPoints[dataPoints[:, bestFeat] == value]
                labelSubset = labels[dataPoints[:, bestFeat] == value]
                self.childNodes[value] = NodeStructure(dpSubset, labelSubset, treeDepth + 1, depthLimit, is_root = False)

        else:
            self.isDiscrete = True
            smallerSubsetDp = dataPoints[dataPoints[:, bestFeat] <= cutoffValue]
            smallerSubsetLabel = labels[dataPoints[:, bestFeat] <= cutoffValue]
            self.childNodes[0] = NodeStructure(smallerSubsetDp, smallerSubsetLabel, treeDepth + 1, depthLimit)
            largerSubsetDp = dataPoints[dataPoints[:, bestFeat] > cutoffValue]
            largerSubsetLabel = labels[dataPoints[:, bestFeat] > cutoffValue]
            self.childNodes[1] = NodeStructure(largerSubsetDp, largerSubsetLabel, treeDepth + 1, depthLimit, is_root = False)



root = NodeStructure(X_train, Y_train, 0, 45)

print("Dtree built!")

def get_all_nodes(root_node):
    all_nodes = list()
    queue = [root_node]
    while len(queue)!=0:
        node = queue.pop(0)
        all_nodes.append(node)
        for child_node in node.children.values():
            if child_node != None: queue.append(child_node)
    return all_nodes

def predict(sample, node):
    if node.LeafNode: 
        return node.result
    else:
        if node.isDiscrete: 
            key = 0 if sample[node.attr]<=node.cutoff else 1
        else:
            key = sample[node.attr]
        if key in node.children: 
            return predict(sample, node.childNodes[key])
        else:
            return node.result

def calculate_accuracy(samples, labels, node):
    preds = [predict(x, node) for x in samples]
    correct_count = sum([a == b for a, b in zip(preds, labels)])
    return correct_count/len(labels)

train_accuracy_list, validation_accuracy_list, test_accuracy_list = list(), list(), list()
node_count_list = list()
all_nodes = get_all_nodes(root)
node_count = len(all_nodes)
while True:
    
    
    node_count_list.append(node_count)
    
    train_acc = calculate_accuracy(X_train, Y_train, root)
    train_accuracy_list.append(train_acc[0])
    
    validation_acc = calculate_accuracy(X_val, Y_val, root)
    validation_accuracy_list.append(validation_acc[0])
    
    test_acc = calculate_accuracy(X_test, Y_test, root)
    test_accuracy_list.append(test_acc[0])
    
    best_node, best_acc = None, -1
    print("******pruning*********")
    for node in all_nodes:
        if node.is_root == False and node.LeafNode == False:
            node.LeafNode = True
            temp_acc = calculate_accuracy(X_val, Y_val, root)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_node = node
            node.LeafNode = False
    if best_acc <= validation_accuracy_list[-1]:
        print("done!")
        break
    else:
        best_node.LeafNode = True
        node_count -= 1

print('plots for depth 45!')
plt.plot(node_count_list, train_accuracy_list, label = 'train vs node count')
plt.plot(node_count_list , validation_accuracy_list, label = 'val vs node count' )
plt.plot(node_count_list, test_accuracy_list, label = 'test vs node count')
plt.xlabel('node count')
plt.ylabel('accuracies')
plt.title('node count vs accuracies for pruning')
plt.legend()
plt.savefig('45.jpg')

