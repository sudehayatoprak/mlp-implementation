#!/usr/bin/env python3
"""
Main script demonstrating both MLP implementations

This script showcases both the from-scratch NumPy implementation and the PyTorch implementation
with proper argument parsing and example usage.

Author: CENG403 Student
Course: CENG403 - Spring 2025
"""

import argparse
import sys
import os
import numpy as np
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.activations import relu, sigmoid, tanh, leakyrelu, elu
from src.mlp_from_scratch import forward_propagation, compute_loss, backward_propagation, update_parameters
from src.pytorch_mlp import (
    MNISTCustomDataset, CustomMLP, train, validate, 
    report_accuracy, plot_loss, compute_confusion_matrix,
    hyperparameter_search
)


def demo_activation_functions():
    """Demonstrate activation functions"""
    print("=" * 50)
    print("ACTIVATION FUNCTIONS DEMO")
    print("=" * 50)
    
    x = np.linspace(-5, 5, 100)
    
    # Test all activation functions
    activations = {
        'ReLU': relu(x),
        'Sigmoid': sigmoid(x),
        'Tanh': tanh(x),
        'Leaky ReLU (α=0.1)': leakyrelu(x, 0.1),
        'ELU (α=1.0)': elu(x, 1.0)
    }
    
    for name, result in activations.items():
        print(f"{name}: min={result.min():.3f}, max={result.max():.3f}, mean={result.mean():.3f}")


def demo_mlp_from_scratch():
    """Demonstrate MLP implementation from scratch"""
    print("\n" + "=" * 50)
    print("MLP FROM SCRATCH DEMO (XOR Problem)")
    print("=" * 50)
    
    # XOR Dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Hyperparameters
    input_size = 2
    hidden_size = 3
    output_size = 1
    learning_rate = 0.01
    num_epochs = 1000
    
    # Initialize parameters
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))
    
    print(f"Training for {num_epochs} epochs...")
    print("Epoch\tLoss\t\tPredictions")
    print("-" * 40)
    
    for epoch in range(num_epochs):
        # Forward propagation
        p_i, cache = forward_propagation(X, W1, b1, W2, b2)
        loss = compute_loss(y, p_i)
        
        # Backward propagation
        gradients = backward_propagation(X, y, cache, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, gradients, learning_rate)
        
        # Print progress
        if epoch % 200 == 0:
            predictions = (p_i > 0.5).astype(int)
            print(f"{epoch}\t{loss:.4f}\t\t{predictions.flatten()}")
    
    # Final results
    p_i, _ = forward_propagation(X, W1, b1, W2, b2)
    predictions = (p_i > 0.5).astype(int)
    accuracy = np.mean(predictions == y) * 100
    
    print(f"\nFinal Results:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Predictions: {predictions.flatten()}")
    print(f"True labels: {y.flatten()}")


def demo_pytorch_mlp(data_dir=None):
    """Demonstrate PyTorch MLP implementation"""
    print("\n" + "=" * 50)
    print("PYTORCH MLP DEMO")
    print("=" * 50)
    
    if data_dir is None:
        print("No data directory provided. Skipping PyTorch demo.")
        print("To run PyTorch demo, provide --data_dir argument with path to MNIST dataset.")
        return
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        print("Please download the MNIST dataset first.")
        return
    
    try:
        from torch.utils.data import DataLoader
        from torchvision import transforms
        import torch.optim as optim
        import torch.nn as nn
        
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Create datasets
        train_dataset = MNISTCustomDataset(os.path.join(data_dir, 'training'), transform=transform)
        val_dataset = MNISTCustomDataset(os.path.join(data_dir, 'validation'), transform=transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CustomMLP(input_size=784, output_size=10, dropout_rate=0.1)
        model.to(device)
        
        # Define loss and optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        print(f"Training on device: {device}")
        print(f"Model architecture:\n{model}")
        
        # Training loop
        num_epochs = 5
        print(f"\nTraining for {num_epochs} epochs...")
        print("Epoch\tTrain Loss\tVal Loss\tVal Acc")
        print("-" * 45)
        
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, loss_function, device)
            val_loss, val_acc = validate(model, val_loader, loss_function, device)
            print(f"{epoch+1}\t{train_loss:.4f}\t\t{val_loss:.4f}\t\t{val_acc:.2f}%")
        
        print(f"\nTraining completed!")
        
    except ImportError as e:
        print(f"PyTorch not available: {e}")
        print("Install PyTorch to run this demo.")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='MLP Implementation Demo')
    parser.add_argument('--demo', choices=['activations', 'scratch', 'pytorch', 'all'], 
                       default='all', help='Which demo to run')
    parser.add_argument('--data_dir', type=str, 
                       help='Path to MNIST dataset directory')
    
    args = parser.parse_args()
    
    print("Multi-layer Perceptron Implementation Demo")
    print("CENG403 - Spring 2025")
    print("=" * 50)
    
    if args.demo in ['activations', 'all']:
        demo_activation_functions()
    
    if args.demo in ['scratch', 'all']:
        demo_mlp_from_scratch()
    
    if args.demo in ['pytorch', 'all']:
        demo_pytorch_mlp(args.data_dir)
    
    print("\n" + "=" * 50)
    print("Demo completed!")


if __name__ == "__main__":
    main()
