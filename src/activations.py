"""
Activation Functions Implementation from Scratch

This module contains implementations of various activation functions and their derivatives
using only NumPy, as part of a Multi-layer Perceptron implementation.

"""

import numpy as np


def relu(x):
    """
    ReLU activation function: max(0, x)
    
    Args:
        x: Input array
        
    Returns:
        ReLU applied element-wise to input
    """
    return np.maximum(0, x)


def sigmoid(x):
    """
    Sigmoid activation function: 1/(1 + e^(-x))
    
    Args:
        x: Input array
        
    Returns:
        Sigmoid applied element-wise to input
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    Tanh activation function: (e^x - e^(-x)) / (e^x + e^(-x))
    
    Args:
        x: Input array
        
    Returns:
        Tanh applied element-wise to input
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def leakyrelu(x, alpha):
    """
    Leaky ReLU activation function
    
    Args:
        x: Input array
        alpha: Negative slope parameter
        
    Returns:
        Leaky ReLU applied element-wise to input
    """
    return np.where(x >= 0, x, alpha * x)


def elu(x, alpha):
    """
    ELU activation function
    
    Args:
        x: Input array
        alpha: Parameter for negative values
        
    Returns:
        ELU applied element-wise to input
    """
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))


# Derivative functions
def relu_derivate(x):
    """
    Derivative of ReLU activation function
    
    Args:
        x: Input array
        
    Returns:
        ReLU derivative applied element-wise to input
    """
    return np.where(x >= 0, 1, 0)


def sigmoid_derivate(x):
    """
    Derivative of Sigmoid activation function
    
    Args:
        x: Input array
        
    Returns:
        Sigmoid derivative applied element-wise to input
    """
    sigmoid_func = 1 / (1 + np.exp(-x))
    return sigmoid_func * (1 - sigmoid_func)


def tanh_derivate(x):
    """
    Derivative of Tanh activation function
    
    Args:
        x: Input array
        
    Returns:
        Tanh derivative applied element-wise to input
    """
    tanh_x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return 1 - (tanh_x ** 2)


def leakyrelu_derivate(x, alpha):
    """
    Derivative of Leaky ReLU activation function
    
    Args:
        x: Input array
        alpha: Negative slope parameter
        
    Returns:
        Leaky ReLU derivative applied element-wise to input
    """
    return np.where(x >= 0, 1, alpha)


def elu_derivate(x, alpha):
    """
    Derivative of ELU activation function
    
    Args:
        x: Input array
        alpha: Parameter for negative values
        
    Returns:
        ELU derivative applied element-wise to input
    """
    return np.where(x >= 0, 1, alpha * np.exp(x))
