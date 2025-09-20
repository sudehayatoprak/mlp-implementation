#!/usr/bin/env python3
"""
Comprehensive training and evaluation script for MLP implementation
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn

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
    """Demonstrate activation functions and save plots"""
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
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, result) in enumerate(activations.items()):
        print(f"{name}: min={result.min():.3f}, max={result.max():.3f}, mean={result.mean():.3f}")
        axes[i].plot(x, result, linewidth=2)
        axes[i].set_title(name)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('f(x)')
    
    # Remove the last empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig('results/activation_functions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Activation functions plot saved to results/activation_functions.png")

def demo_mlp_from_scratch():
    """Demonstrate MLP implementation from scratch with XOR problem"""
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
    
    losses = []
    
    for epoch in range(num_epochs):
        # Forward propagation
        p_i, cache = forward_propagation(X, W1, b1, W2, b2)
        loss = compute_loss(y, p_i)
        losses.append(loss)
        
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
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('XOR Problem - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/xor_training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("XOR training loss plot saved to results/xor_training_loss.png")

def train_pytorch_mlp(data_dir, num_epochs=10):
    """Train PyTorch MLP and generate comprehensive results"""
    print("\n" + "=" * 50)
    print("PYTORCH MLP TRAINING")
    print("=" * 50)
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return
    
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
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"\nTraining for {num_epochs} epochs...")
    print("Epoch\tTrain Loss\tVal Loss\tVal Acc")
    print("-" * 45)
    
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, loss_function, device)
        val_loss, val_acc = validate(model, val_loader, loss_function, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"{epoch+1}\t{train_loss:.4f}\t\t{val_loss:.4f}\t\t{val_acc:.2f}%")
    
    print(f"\nTraining completed!")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training curves saved to results/training_curves.png")
    
    # Generate confusion matrix
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    confusion_matrix = compute_confusion_matrix(all_predictions, all_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i, j in np.ndindex(confusion_matrix.shape):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                ha="center", va="center",
                color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved to results/confusion_matrix.png")
    
    # Save results to file
    with open('results/training_results.txt', 'w') as f:
        f.write("PyTorch MLP Training Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final Validation Loss: {val_losses[-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%\n")
        f.write(f"Best Validation Accuracy: {max(val_accuracies):.2f}%\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix))
        f.write("\n\n")
        
        f.write("Per-class accuracy:\n")
        for i in range(10):
            class_correct = confusion_matrix[i, i]
            class_total = confusion_matrix[i, :].sum()
            accuracy = class_correct / class_total * 100
            f.write(f"Class {i}: {accuracy:.2f}% ({class_correct}/{class_total})\n")
    
    print("Training results saved to results/training_results.txt")

def main():
    """Main function"""
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    print("Multi-layer Perceptron Implementation Demo")
    print("CENG403 - Spring 2025")
    print("=" * 50)
    
    # Demo activation functions
    demo_activation_functions()
    
    # Demo MLP from scratch
    demo_mlp_from_scratch()
    
    # Train PyTorch MLP
    train_pytorch_mlp('data', num_epochs=10)
    
    print("\n" + "=" * 50)
    print("All demos completed! Check the results/ directory for outputs.")
    print("=" * 50)

if __name__ == "__main__":
    main()
