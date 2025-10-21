from utils import get_one_hot_array
from utils import best_attribute
import matplotlib.pyplot as plt
import numpy as np

X, y, attributes, type = get_one_hot_array("d-tree/train.csv")
X_val, Y_val, attributes, type = get_one_hot_array('d-tree/validation.csv')
X_test, Y_test, attributes, type = get_one_hot_array('d-tree/test.csv')


class DtreeNode:
    def __init__(self, dataPoints, labels):
        self.dataPoints = dataPoints
        self.labels = labels
        self.children = dict()
        self.leafNode = False
        self.bestFeat = None
        self.cutoff = None
        self.result = None

    
    def growTree(self, TreeDepth, DepthLimit):
        positive_labels = sum(self.labels)
        negative_labels = len(self.labels) - positive_labels

        if positive_labels == 0 or negative_labels == 0 or TreeDepth == DepthLimit:
            self.leafNode = True
            self.result = 1 if positive_labels > negative_labels else 0
            return
        
        self.result = 1 if positive_labels > negative_labels else 0
        bestFeat, cutoff = best_attribute(self.dataPoints, self.labels, [])
        self.bestFeat = bestFeat
        self.cutoff = cutoff
        left_child_dp = self.dataPoints[self.dataPoints[:, bestFeat] <= cutoff]
        left_child_labels = self.labels[self.dataPoints[:, bestFeat] <= cutoff]
        self.children[0] = DtreeNode(left_child_dp, left_child_labels)
        self.children[0].growTree(TreeDepth + 1, DepthLimit)

        right_child_dp = self.dataPoints[self.dataPoints[:, bestFeat] > cutoff]
        right_child_labels = self.labels[self.dataPoints[:, bestFeat] > cutoff]
        self.children[1] = DtreeNode(right_child_dp, right_child_labels)
        self.children[1].growTree(TreeDepth + 1, DepthLimit)

def check_accuracy(dataset, actualLabels, root):
    predictedLabels = []

    for data in dataset:
        current_node = root  
        while not current_node.leafNode:
            if data[current_node.bestFeat] <= current_node.cutoff:  
                if 0 in current_node.children and current_node.children[0] is not None:
                    current_node = current_node.children[0]
                else:
                    break
            else:
                if 1 in current_node.children and current_node.children[1] is not None:
                    current_node = current_node.children[1]
                else:
                    break

        predictedLabels.append(current_node.result)

    successCount = sum(1 for i in range(len(predictedLabels)) if predictedLabels[i] == actualLabels[i])
    totalAccuracy = successCount / len(actualLabels) * 100

    zeroClassAccuracy = sum(1 for correct, estimated in zip(actualLabels, predictedLabels) if correct == 0 and estimated == 0)
    zeroClassTotal = np.sum(actualLabels == 0)
    oneClassAccuracy = sum(1 for correct, estimated in zip(actualLabels, predictedLabels) if correct == 1 and estimated == 1)
    oneClassTotal = np.sum(actualLabels == 1)

    class_0_accuracy = zeroClassAccuracy / zeroClassTotal if zeroClassTotal > 0 else 0.0
    class_1_accuracy = oneClassAccuracy / oneClassTotal if oneClassTotal > 0 else 0.0

    return totalAccuracy, class_0_accuracy, class_1_accuracy



depths = [15, 25, 35, 45]

train_accuracy = []
val_accuracy = []
test_accuracy = []
for depth in depths:
    root = DtreeNode(X, y)
    root.growTree(0, depth)
    accuracy_train, class_0_accuracy_train, class_1_accuracy_train = check_accuracy(X, y, root)
    accuracy_val, class_0_accuracy_val, class_1_accuracy_val = check_accuracy(X_val, Y_val, root)
    accuracy_test, class_0_accuracy_test, class_1_accuracy_test = check_accuracy(X_test, Y_test, root)
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
plt.savefig('b.jpg')
