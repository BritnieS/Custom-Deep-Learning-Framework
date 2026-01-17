# Convolutional Neural Networks (CNN) from Scratch

## 📌 Project Overview

This project extends a custom-built Deep Learning framework to support **Convolutional Neural Networks (CNNs)**. The goal was to understand the mathematical internals of CNNs by implementing the forward and backward passes of convolutional and pooling layers manually, without relying on high-level libraries like PyTorch or Keras.

This implementation includes advanced initialization schemes, adaptive optimizers, and support for multi-dimensional tensor processing.

## 📂 Project Structure & Implementation Details

### 1. Advanced Initializers (`Layers/Initializers.py`)

Implemented various initialization strategies to ensure stable convergence for non-convex optimization problems:

* **Constant:** Initializes weights with a fixed value.
* **UniformRandom:** Standard uniform distribution .
* **Xavier (Glorot):** Optimized for Sigmoid/Tanh activations.
* **He Initialization:** Optimized for ReLU activations to prevent vanishing gradients.

### 2. Advanced Optimizers (`Optimization/Optimizers.py`)

Implemented sophisticated optimization algorithms to speed up convergence:

* **SGD with Momentum:** Adds a velocity term to standard Stochastic Gradient Descent to navigate ravines and reduce oscillation.
* **Adam:** Adaptive Moment Estimation using first () and second () moments of gradients.

### 3. Layer Architectures

#### **Flatten Layer** (`Layers/Flatten.py`)

* Reshapes multi-dimensional tensors (e.g., `[Batch, Channels, Height, Width]`) into a 1D feature vector for Fully Connected layers.


* Implemented both `forward` and `backward` reshaping logic.

#### **Convolutional Layer** (`Layers/Conv.py`)

The core of the project. Implemented trainable 1D and 2D convolutions:

* **Forward Pass:** Handles input layouts (`Batch, Channel, Y, X`), stride application, and "same" padding (zero-padding) to preserve spatial dimensions.
* **Backward Pass:** Computes gradients with respect to weights, bias, and input error.
* **Features:** Supports 1D and 2D convolution shapes and arbitrary stride configurations.

#### **Pooling Layer** (`Layers/Pooling.py`)

Implemented **Max-Pooling** to reduce dimensionality and introduce translation invariance.

* **Forward Pass:** Downsamples the input using a "valid" padding strategy (no padding).
* **Backward Pass:** Propagates error only to the indices that were selected as the "max" during the forward pass.

## 💻 Tech Stack

* **Language:** Python 3
* 
**Libraries:** NumPy, SciPy (for n-dimensional convolution operations).


* **Framework:** Custom implementation (No ML libraries used).

## 🚀 How to Run & Test

The project includes a comprehensive test suite to verify the mathematical correctness of gradients and layer logic.

**Run the full test suite:**

```bash
python3 NeuralNetworkTests.py

```

**Run specific component tests:**

```bash
[cite_start]python3 NeuralNetworkTests.py TestInitializers  # Test Init schemes [cite: 1]
[cite_start]python3 NeuralNetworkTests.py TestOptimizers2   # Test Adam/Momentum [cite: 2]
[cite_start]python3 NeuralNetworkTests.py TestConv          # Test Convolution logic [cite: 5]
[cite_start]python3 NeuralNetworkTests.py TestPooling       # Test Pooling logic [cite: 6]

```

**Check for Bonus Objective completion:**

```bash
python3 NeuralNetworkTests.py Bonus

```

## 📜 Assignment Context

This exercise is part of the "Deep Learning" course, aimed at providing a low-level understanding of neural network calculus and architecture. The focus is on efficiency trade-offs, memory management, and tensor operations.

---

*Author: Britnie Sinthuja*
