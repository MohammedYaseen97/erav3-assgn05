# Fast MNIST CNN Implementation
![CI](https://github.com/MohammedYaseen97/erav3-assgn05/actions/workflows/model_tests.yml/badge.svg)

A lightweight and efficient Convolutional Neural Network (CNN) implementation for the MNIST digit classification task, optimized for both performance and model size.

## 🎯 Project Goals

- Achieve 95%+ accuracy on MNIST digit classification
- Maintain a lightweight architecture (<25,000 parameters)
- Provide easy-to-use training and testing interfaces

## 🛠️ Technical Requirements

- Python 3.x
- PyTorch
- torchvision
- pytest (for running tests)

## 📦 Installation

```bash
pip install -r requirements.txt
```
## 🧪 Testing

The project includes comprehensive tests to verify both model architecture and performance:

```bash
pytest test_mnist_model.py
```

Test suite includes:
- Model size verification (ensures <25k parameters)
- Training accuracy validation (ensures >95% accuracy)

## 🏗️ Project Structure

.
├── mnist_cnn.py # CNN model implementation
├── test_mnist_model.py # Test suite
├── requirements.txt # Project dependencies
└── .github/workflows # CI/CD configuration

## 🔍 Model Architecture

The `FastMNISTCNN` is a compact CNN architecture specifically designed for MNIST digit classification, balancing efficiency with performance.

## 🚀 CI/CD

The project includes GitHub Actions workflows for automated testing and validation.