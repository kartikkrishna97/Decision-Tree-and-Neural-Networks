from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import time 
from utils import preprocess_data
from functools import reduce

train_path = "train.csv"
val_path = 'val.csv'
test_csv = 'test.csv'

X_train, Y_train, type, attributes =preprocess_data(train_path)
X_val, Y_val, type, attributes = preprocess_data(val_path)
X_test, Y_test, type, attributes = preprocess_data(val_path)


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
                self.childNodes[value] = NodeStructure(dpSubset, labelSubset, treeDepth + 1, depthLimit)

        else:
            self.isDiscrete = True
            smallerSubsetDp = dataPoints[dataPoints[:, bestFeat] <= cutoffValue]
            smallerSubsetLabel = labels[dataPoints[:, bestFeat] <= cutoffValue]
            self.childNodes[0] = NodeStructure(smallerSubsetDp, smallerSubsetLabel, treeDepth + 1, depthLimit)
            largerSubsetDp = dataPoints[dataPoints[:, bestFeat] > cutoffValue]
            largerSubsetLabel = labels[dataPoints[:, bestFeat] > cutoffValue]
            self.childNodes[1] = NodeStructure(largerSubsetDp, largerSubsetLabel, treeDepth + 1, depthLimit)


def check_accuracy(dataset, actualLabels, rootNode):
    predictedLabels = []
    successCount = 0

    for data in dataset:
        workingNode = rootNode
        while not workingNode.isLeafNode:
            if workingNode.isDiscrete:
                if data[workingNode.feature] <= workingNode.cutoff:
                    if workingNode.childNodes[0] is not None:
                        workingNode = workingNode.childNodes[0]
                    else:
                        break
                else:
                    if workingNode.childNodes[1] is not None:
                        workingNode = workingNode.childNodes[1]
                    else:
                        break
            else:
                if data[workingNode.feature] in workingNode.childNodes and workingNode.childNodes[data[workingNode.feature]] is not None:
                    workingNode = workingNode.childNodes[data[workingNode.feature]]
                else:
                    break
        predictedLabels.append(workingNode.result)

    for i in range(len(predictedLabels)):
        if predictedLabels[i] == actualLabels[i]:
            successCount += 1

    totalAccuracy = successCount / len(actualLabels) * 100

    zeroClassAccuracy = sum(1 for correct, estimated in zip(actualLabels, predictedLabels) if correct == 0 and estimated == 0)
    zeroClassTotal = np.sum(actualLabels==0)
    oneClassAccuracy = sum(1 for correct, estimated in zip(actualLabels, predictedLabels) if correct == 1 and estimated == 1)
    oneClassTotal = np.sum(actualLabels==1)

    class_0_accuracy = zeroClassAccuracy / zeroClassTotal if zeroClassTotal > 0 else 0.0
    class_1_accuracy = oneClassAccuracy / oneClassTotal if oneClassTotal > 0 else 0.0

    return totalAccuracy, class_0_accuracy, class_1_accuracy

depths = [15, 25, 35, 45]





train_accuracy = []
val_accuracy = []
test_accuracy = []
for depth in depths:
    root = NodeStructure(X_train, Y_train, 0, depth)
    accuracy_train, class_0_accuracy_train, class_1_accuracy_train = check_accuracy(X_train, Y_train, root)
    accuracy_val, class_0_accuracy_val, class_1_accuracy_val = check_accuracy(X_val, Y_val, root)
    accuracy_test, class_0_accuracy_test, class_1_accuracy_test = check_accuracy(X_val, Y_val, root)
    print(f"At depth {depth}, accuracy for predicting class 0 on train is {100*class_0_accuracy_train} ")
    print(f"At depth {depth}, accuracy for predicting class 1 on train is {100*class_1_accuracy_train} ")
    print(f"At depth {depth}, accuracy for predicting class 0 on val is {100*class_0_accuracy_val} ")
    print(f"At depth {depth}, accuracy for predicting class 1 on val is {100*class_1_accuracy_val} ")
    print(f"At depth {depth}, accuracy for predicting class 0 on test is {100*class_0_accuracy_test} ")
    print(f"At depth {depth}, accuracy for predicting class 1 on test is {100*class_1_accuracy_test} ")
    train_accuracy.append(accuracy_train)
    val_accuracy.append(accuracy_val)
    test_accuracy.append(accuracy_test)
    print(accuracy_train)
    print(accuracy_val)
    print(accuracy_test)


plt.plot(depths, train_accuracy, label = 'train accuracy')
plt.plot(depths, val_accuracy, label = 'val accuracy')
plt.plot(depths, test_accuracy, label = 'test accuracy')
plt.xlabel('depth')
plt.ylabel('accuracy')
plt.title('accuracy vs depth plot for train test and val')
plt.legend()
plt.savefig('a.jpg')