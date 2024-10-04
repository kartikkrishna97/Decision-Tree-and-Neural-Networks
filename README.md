# Decision Trees and Neural Networks Implementations

This repository contains custom implementations of Decision Trees and Neural Networks, alongside comparisons with their scikit-learn counterparts. Each folder provides a specific set of experiments and implementations for various model architectures.

## Folder Structure

### Decision Tree

The **Decision Tree** folder includes the following implementations and experiments:

- **a) Decision Tree with Ordinal Encoding**  
  Implementation of a Decision Tree using ordinal encoding. Experiments are conducted to evaluate model performance with varying tree depths.

- **b) Decision Tree with One-Hot Encoding**  
  Implementation of a Decision Tree using one-hot encoding. Experiments are conducted to evaluate model performance with varying tree depths.

- **c) Decision Tree with Pruning**  
  Decision Tree implementation with pruning, where pruning decisions are based on node accuracy.

- **d) Comparison with scikit-learn Decision Tree**  
  Comparison of the custom Decision Tree implementation against the scikit-learn `DecisionTreeClassifier`. Includes experiments with pruning using `ccp_alpha`.

- **e) Comparison with scikit-learn Random Forests**  
  Comparison of the custom Decision Tree implementation against scikit-learn's `RandomForestClassifier` to assess performance differences between single trees and ensemble models.

### Neural Network

The **Neural Network** folder includes the following implementations and experiments:

- **a) Neural Network with Custom Backpropagation**  
  Implementation of a Neural Network from scratch, including backpropagation. Experiments with varying network depths are provided.

- **b) Comparison with scikit-learn Neural Network**  
  A comparison between the custom Neural Network implementation and scikit-learn’s `MLPClassifier` to evaluate performance and scalability.

## How to Run the Code

To run the experiments, navigate to the appropriate folder (Decision Tree or Neural Network) and execute the desired script:

```bash
python a/b/c/d/e.py
