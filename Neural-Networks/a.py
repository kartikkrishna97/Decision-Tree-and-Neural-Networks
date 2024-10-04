import numpy as np 
import sys
import pdb
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import math
from tqdm import tqdm


learning_rate = 0.01
batch_size = 32

def categorical_cross_entropy(true_labels, predicted_probs):
    
    epsilon = 1e-15
    predicted_probs = np.maximum(epsilon, predicted_probs)

    
    loss = -np.sum(true_labels * np.log(predicted_probs))

    # Optionally, you can calculate the average loss over a batch
    num_samples = true_labels.shape[0]
    average_loss = loss / num_samples

    return  average_loss


def softmax(x):
    unnormalized_probs = np.exp(x)
    denominator = np.sum(unnormalized_probs, axis=1, keepdims=True)
    softmax_probs = unnormalized_probs / denominator
    return softmax_probs



def batch_stable_softmax(x):
    # Subtract the maximum value along each row (axis=1) to avoid overflow
    max_x = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - max_x)
    
    # Sum along each row to get the denominator
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    
    # Calculate the softmax for each data point in the batch
    softmax = exp_x / sum_exp_x
    
    return softmax



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def ReLU_activation(z):
    return np.maximum(0, z)

def ReLU_gradient(x):
    return (x > 0) * 1

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    
    return np.where(x >= 0, 1, alpha * np.exp(x))

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

label_encoder = OneHotEncoder(sparse_output = False)
label_encoder.fit(np.expand_dims(Y_train, axis = -1))



y_train_onehot = label_encoder.transform(np.expand_dims(Y_train, axis = -1))
y_test_onehot = label_encoder.transform(np.expand_dims(Y_test, axis = -1))


Features = X_train.shape[1]
hidden_neurons = [512, 256]
num_classes = 5

def make_layers(features:int, hidden_neurons:list, num_classes:int):
    layers = [features]+hidden_neurons+[num_classes]
    weights = []
    for i in range(len(layers)-1):
        if i == len(layers)-2:
            weights.append(np.random.normal(0,1,(layers[i]+1,layers[i+1])))
            

        else:
            t = np.random.normal(0,1,(layers[i]+1,layers[i+1]))
            # t = np.random.uniform(-1*epsilon, 1*epsilon, (layers[i]+1,layers[i+1]))
            ones = np.ones((t.shape[0],1))
            s = np.hstack((ones, t))
            weights.append(s)

    return weights


W = make_layers(Features,hidden_neurons,num_classes)





def forward_prop(weights:list, input:list):
  '''
    Args:
        weights: weight for each layer [[features x hidden_neurons], [hidden_neurons x num_classes]]
        input:   size of the training data [num samples x features]
 
    '''
  ones =  np.ones(input.shape[0])
  new_input = np.column_stack((ones, input))
  activations = [new_input]
  for i in range((len(hidden_neurons)+1)):
      if i == len(hidden_neurons):
        y = batch_stable_softmax(np.dot(activations[i], weights[i]))
        activations.append(y)

      else:
          t = ReLU_activation(np.dot(activations[i], weights[i]))
          activations.append(t)
  return activations

# activations = forward_prop(W, X_train)



def backprop(forward_weights:list, target_value:list, weights:list):
    '''
    Args:
        forward_weights: forward prop weights for each layer [[batchsize x features][batchsize x hidden_neurons], [batchsize x num_classes]]
        target values: one hot encoded vector of ground truth labels [batchsize x num_classes]
        weights: weight for each layer [[features x hidden_neurons], [hidden_neurons x num_classes]]
 
    '''
    

    error_last = forward_weights[-1]-target_value
    delta = []
    delta.append(error_last)
    gradient = []
    j = 0
    for i in range(len(weights)-1,-1,-1):
        if i == len(weights)-1:
            x = forward_weights[i].T
            grad_temp = np.matmul(x,delta[j])
            gradient.append(grad_temp/batch_size)
        else:
            der = ReLU_gradient(forward_weights[i+1])
            temp = np.dot(delta[j], weights[i+1].T)
            temp_new = der*temp
            delta.append(temp_new)
            grad_temp = np.dot(forward_weights[i].T,temp_new)
            gradient.append(grad_temp/batch_size)
            j+=1

    return gradient



def update_weights(gradients:list, weights:list, lr:float):
    '''
    Args:
        gradients: gradients calcluated for each layer [[features x hidden_neurons], [hidden_neurons x num_classes]]
        target values: one hot encoded vector of ground truth labels [batchsize x num_classes]
        weights: weight for each layer [[features x hidden_neurons],.........,[hidden_neurons x num_classes]]
 
    '''
    grads = gradients[::-1]
    updated_weights = [] 
    for i in range(len(weights)):
          updated_weights.append(weights[i]-lr*grads[i])
          
    return updated_weights
      

def iterate_batches(X:list,Y:list, batch_size:int):
    '''
    X: Training data [batch_size x features]
    Y: one hot encoded labels [batch_size x num_classes]
    batch_size: number of batches to divide the training data into
    '''
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], Y[i:i+batch_size]

loss_total = []
count = 0
train_acc_total = []
test_acc = []
f1_score_train = []
for i in (range(500)):
    if i!=0:
        learning_rate = learning_rate/math.sqrt(i)

    loss_batch = []
    acc_batch = []
    f1_score_batch = []
    for x,y in iterate_batches(X_train,y_train_onehot, batch_size):
        activations = forward_prop(W,x)
        gradients = backprop(activations,y,W)
        W = update_weights(gradients, W, learning_rate)
        loss_batch.append(categorical_cross_entropy(y,activations[-1]))
        softmax_output = activations[-1]
        predicted_labels = np.argmax(softmax_output, axis=1)
        class_labels = np.argmax(y, axis=1)
        accuracy = accuracy_score(class_labels, predicted_labels)
        f1_score_new = f1_score(class_labels, predicted_labels,average='weighted' )
        f1_score_batch.append(f1_score_new)
        acc_batch.append(accuracy)
    
    loss = sum(loss_batch)/len(loss_batch)
    accuracy = 100*(sum(acc_batch)/len(acc_batch))
    f1_score_1 = 100*sum(f1_score_batch)/len(f1_score_batch)
    f1_score_train.append(f1_score_1)

    

    
    print(f"loss is {loss}")
    print(f"train accuracy is {accuracy}")


    loss_total.append(loss)
    train_acc_total.append(accuracy)




    if i > 0 and abs(loss_total[i] - loss_total[i - 1]) <= 1e-3:

        count+=1
        if count==5:
            break

softmax_output = activations[-1]
predicted_labels_train = np.argmax(softmax_output, axis=1)+1

activations_test = forward_prop(W,X_test)

softmax_output_test = activations_test[-1]
predicted_labels_test = np.argmax(softmax_output_test, axis=1)+1

test_accuracy = accuracy_score(Y_test,predicted_labels_test)
print(f"test_accuracy  for depth of {len(hidden_neurons)} layers is {test_accuracy}")
print(f"train accuracy  for depth of {len(hidden_neurons)} layers is {train_acc_total[-1]}")
print(f"f1 score test  for depth of {len(hidden_neurons)} layers is  {f1_score(Y_test,predicted_labels_test,average='weighted')}")
print(f"f1 score for depth of {len(hidden_neurons)} layers is {f1_score_train[-1]}")
print(f"precision for  {len(hidden_neurons)} layers is {precision_score(Y_test,predicted_labels_test,average='weighted')}")
print(f"recall for  of {len(hidden_neurons)} layers is {recall_score(Y_test,predicted_labels_test,average='weighted')}")

f1_scores = f1_score(Y_test, predicted_labels_test, average=None)

# Get the class labels
class_labels = [1, 2, 3, 4, 5]

# Print class-wise F1 scores
for i, label in enumerate(class_labels):
    print(f'F1 Score for Class {label}: {f1_scores[i]}')

precision_scores = precision_score(Y_test, predicted_labels_test, average=None)
recall_scores = recall_score(Y_test, predicted_labels_test, average=None)
# Get the class labels
class_labels = [1, 2, 3, 4, 5]

# Print class-wise F1 scores
for i, label in enumerate(class_labels):
    print(f"precision score for Class {label}: {precision_scores[i]}")

for i, label in enumerate(class_labels):
    print(f'recall Score for Class {label}: {recall_scores[i]}')

