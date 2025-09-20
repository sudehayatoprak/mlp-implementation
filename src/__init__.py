"""
MLP Implementation Package

This package contains implementations of Multi-layer Perceptrons both from scratch
and using PyTorch for educational purposes.

Modules:
- activations: Activation functions and their derivatives
- mlp_from_scratch: Complete MLP implementation using only NumPy
- pytorch_mlp: PyTorch-based MLP implementation for MNIST classification
"""

from .activations import (
    relu, sigmoid, tanh, leakyrelu, elu,
    relu_derivate, sigmoid_derivate, tanh_derivate, 
    leakyrelu_derivate, elu_derivate
)

from .mlp_from_scratch import (
    forward_propagation, compute_loss, 
    backward_propagation, update_parameters
)

from .pytorch_mlp import (
    MNISTCustomDataset, CustomMLP, train, validate, 
    report_accuracy, plot_loss, compute_confusion_matrix,
    hyperparameter_search
)

__version__ = "1.0.0"
__author__ = "CENG403 Student"
