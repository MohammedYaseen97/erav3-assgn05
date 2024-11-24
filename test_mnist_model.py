import torch
import pytest
from mnist_cnn import FastMNISTCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import time

@pytest.fixture
def model():
    return FastMNISTCNN()

def test_model_size(model):
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Model should have less than 25k parameters
    assert total_params <= 25_000, f"Model has {total_params} parameters, exceeding 25k limit"

def test_model_training_accuracy():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Model setup
    model = FastMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for one epoch
    model.train()
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    assert accuracy >= 95, f"Training accuracy is {accuracy:.2f}%, below 95% threshold"

def test_gradient_check():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    # Model setup
    model = FastMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Get a single batch
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    for param in model.parameters():
        assert param.grad is not None, "Gradient not computed for some parameters"

def test_model_serialization():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model setup
    model = FastMNISTCNN().to(device)
    
    # Save the model
    torch.save(model.state_dict(), 'temp_model.pth')
    
    # Load the model
    loaded_model = FastMNISTCNN().to(device)
    loaded_model.load_state_dict(torch.load('temp_model.pth'))
    
    # Check if the parameters are the same
    for param1, param2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(param1, param2), "Model parameters do not match after loading"

def test_inference_time():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Model setup
    model = FastMNISTCNN().to(device)
    model.eval()
    
    # Measure inference time
    images, _ = next(iter(test_loader))
    images = images.to(device)
    
    start_time = time.time()
    with torch.no_grad():
        _ = model(images)
    end_time = time.time()
    
    inference_time = end_time - start_time
    assert inference_time < 0.01, f"Inference time is too high: {inference_time:.4f} seconds"