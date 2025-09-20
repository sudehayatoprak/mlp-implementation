"""
Multi-layer Perceptron Implementation from Scratch

This module contains a complete implementation of a two-layer MLP for binary classification
using only NumPy. It includes forward propagation, backward propagation, and parameter updates.

Author: CENG403 Student
Course: CENG403 - Spring 2025
"""

import numpy as np
from .activations import relu, sigmoid


def forward_propagation(X, W1, b1, W2, b2):
    """
    Forward propagation through the two-layer MLP
    
    Architecture:
    z1 = W1^T * X^T + b1^T
    h = ReLU(z1)
    z2 = W2^T * h + b2
    p_i = sigmoid(z2)
    
    Args:
        X: Input data (n_samples, input_size)
        W1: Weight matrix for first layer (input_size, hidden_size)
        b1: Bias vector for first layer (hidden_size,)
        W2: Weight matrix for second layer (hidden_size, output_size)
        b2: Bias vector for second layer (output_size,)
        
    Returns:
        p_i: Predictions (n_samples, output_size)
        cache: Tuple containing intermediate values for backpropagation
    """
    # First layer: Linear transformation
    Z1 = np.dot(W1.T, X.T) + b1.T  # X: (n, input_size), W1: (input_size, hidden_size), b1: (hidden_size,)
    
    # Apply ReLU activation
    h = np.maximum(Z1, 0)  # ReLU: element-wise
    
    # Second layer: Linear transformation
    Z2 = np.matmul(W2.T, h) + b2  # h: (n, hidden_size), W2: (hidden_size, output_size), b2: (output_size,)
    
    # Apply Sigmoid activation
    p_i = 1 / (1 + np.exp(Z2 * (-1)))  # Sigmoid: element-wise
    p_i = p_i.T
    
    cache = (Z1, h, Z2, p_i)
    return p_i, cache


def compute_loss(y, p_i):
    """
    Compute binary cross-entropy loss
    
    L_BCE = -(1/N) * sum(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))
    
    Args:
        y: True labels (n_samples, 1)
        p_i: Predictions (n_samples, 1)
        
    Returns:
        loss: Binary cross-entropy loss
    """
    y = y.reshape(-1, 1)
    N = len(p_i)
    result = -np.sum(y * np.log(p_i) + (1 - y) * np.log(1 - p_i)) / N
    return result


def backward_propagation(X, y, cache, W2):
    """
    Backward propagation to compute gradients
    
    Args:
        X: Input data (n_samples, input_size)
        y: True labels (n_samples, 1)
        cache: Tuple containing intermediate values from forward pass
        W2: Weight matrix for second layer
        
    Returns:
        gradients: Tuple containing gradients (dW1, db1, dW2, db2)
    """
    Z1, h, Z2, p_i = cache
    N = X.shape[0]
    
    # Compute gradients for the second layer
    dZ2 = (p_i.T - y.T)  # shape: (1, n)
    dW2 = np.dot(h, dZ2.T) / N  # (hidden_size, n) @ (n, 1) = (hidden_size, 1)
    db2 = np.sum(dZ2, axis=1, keepdims=True) / N  # (1, 1)
    
    # Compute gradients for the first layer
    dZ1 = np.dot(dZ2.T, W2.T) * (Z1.T > 0)  # → (n, hidden_size)
    dW1 = np.dot(X.T, dZ1) / N  # (input_size, n) @ (n, hidden_size) = (input_size, hidden_size)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / N
    
    gradients = (dW1, db1, dW2, db2)
    return gradients


def update_parameters(W1, b1, W2, b2, gradients, learning_rate):
    """
    Update model parameters using gradient descent
    
    Args:
        W1, b1, W2, b2: Current model parameters
        gradients: Tuple containing gradients (dW1, db1, dW2, db2)
        learning_rate: Learning rate for gradient descent
        
    Returns:
        Updated parameters (W1, b1, W2, b2)
    """
    dW1, db1, dW2, db2 = gradients
    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    return W1, b1, W2, b2
