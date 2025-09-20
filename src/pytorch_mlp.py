"""
PyTorch Multi-layer Perceptron Implementation

This module contains PyTorch implementations for MNIST classification including
custom dataset, model architecture, training, and evaluation functions.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


class MNISTCustomDataset(Dataset):
    """
    Custom dataset class for loading MNIST images from directory structure
    """
    
    def __init__(self, data_dir, transform=None):
        """
        Initialize dataset
        
        Args:
            data_dir: Path to dataset directory
            transform: Transformations to apply to images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load image file paths and labels
        for label in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label)
            for image in os.listdir(label_path):
                image_path = os.path.join(label_path, image)
                self.images.append(image_path)
                self.labels.append(int(label))
    
    def __len__(self):
        """Return number of samples in dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get item by index
        
        Args:
            idx: Index of the item
            
        Returns:
            image: PIL Image or transformed tensor
            label: Integer label
        """
        image_path = self.images[idx]
        image = Image.open(image_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CustomMLP(nn.Module):
    """
    Custom Multi-layer Perceptron for MNIST classification
    """
    
    def __init__(self, input_size, output_size, dropout_rate=0.5):
        """
        Initialize MLP architecture
        
        Args:
            input_size: Size of input features
            output_size: Number of output classes
            dropout_rate: Dropout probability
        """
        super(CustomMLP, self).__init__()
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


def train(model, train_loader, optimizer, loss_function, device):
    """
    Train the model for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        loss_function: Loss function
        device: Device to run on (CPU/GPU)
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, loss_function, device):
    """
    Validate the model
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        loss_function: Loss function
        device: Device to run on (CPU/GPU)
        
    Returns:
        Average validation loss and accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            predictions = model(images)
            loss = loss_function(predictions, labels)
            total_loss += loss.item()
            
            correct += (predictions.argmax(1) == labels).type(torch.float).sum().item()
            total_samples += labels.size(0)
    
    accuracy = correct / total_samples * 100
    return total_loss / len(val_loader), accuracy


def report_accuracy(data_loader, model, device='cpu'):
    """
    Report accuracy on a dataset
    
    Args:
        data_loader: Data loader
        model: PyTorch model
        device: Device to run on (CPU/GPU)
        
    Returns:
        Accuracy percentage
    """
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = (correct / total) * 100
    return accuracy


def plot_loss(train_losses, val_losses):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Epoch')
    plt.legend()
    plt.show()


def compute_confusion_matrix(predictions, true_labels, num_classes=10):
    """
    Compute confusion matrix using only NumPy
    
    Args:
        predictions: List of predicted labels
        true_labels: List of true labels
        num_classes: Number of classes
        
    Returns:
        Confusion matrix as numpy array
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for pred, true in zip(predictions, true_labels):
        confusion_matrix[pred, true] += 1
    
    return confusion_matrix


def hyperparameter_search(learning_rates, dropout_rates, train_loader, val_loader, 
                         input_size, output_size, num_epochs, device):
    """
    Perform grid search over hyperparameters
    
    Args:
        learning_rates: List of learning rates to try
        dropout_rates: List of dropout rates to try
        train_loader: Training data loader
        val_loader: Validation data loader
        input_size: Input feature size
        output_size: Number of output classes
        num_epochs: Number of training epochs
        device: Device to run on
        
    Returns:
        Dictionary with results for each hyperparameter combination
    """
    results = {}
    loss_function = nn.CrossEntropyLoss()
    weight_decay = 1e-4
    
    for lr in learning_rates:
        for dr in dropout_rates:
            print(f"\nTraining with learning rate = {lr} and dropout rate = {dr}")
            
            model = CustomMLP(input_size, output_size, dropout_rate=dr)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            best_val_accuracy = 0.0
            best_val_loss = float('inf')
            
            for epoch in range(num_epochs):
                train_loss = train(model, train_loader, optimizer, loss_function, device)
                val_loss, val_acc = validate(model, val_loader, loss_function, device)
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
                
                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    best_val_loss = val_loss
            
            results[(lr, dr)] = {
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_accuracy
            }
    
    return results
