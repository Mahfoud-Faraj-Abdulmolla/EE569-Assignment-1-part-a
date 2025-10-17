# EE569-Assignment-1-part-a
# Assignment 1 – Task 4: Effect of Batch Size on Training Loss

## Objective
Investigate how batch size affects the training of a logistic regression model, focusing on training loss and convergence behavior.

## Implementation
- Logistic regression implemented using **NumPy** with batch-based gradient descent.
- Forward pass: `z = X @ W.T + b`
- Activation: `sigmoid(z) = 1 / (1 + exp(-z))`
- Loss: Binary cross-entropy  
  `L = -1/N * Σ[y*log(y_hat) + (1-y)*log(1-y_hat)]`

## Experimental Setup
- Dataset: 500 samples, 2 features, random binary labels
- Learning rate: 0.01
- Epochs: 10000
- Batch sizes tested: 1, 2, 10, 50, 250, 500

## Results
![Training Loss vs Epochs](path_to_your_plot.png)

**Observations:**
- Small batches (1,2) → noisy updates, slower convergence
- Medium batches (10,50) → faster convergence, smoother loss
- Large batches (250,500) → stable but slower per-epoch convergence

**Trade-off:**
- Small batches: faster updates, higher gradient noise
- Large batches: smoother updates, slower progress per epoch

## Conclusion
- Batch size significantly affects training dynamics.
- Small batches → unstable but quick updates
- Large batches → stable but computationally heavier
- Medium batch sizes (e.g., 50) balance speed and stability.

