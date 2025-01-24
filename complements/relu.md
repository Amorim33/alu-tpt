The **RELU (Rectified Linear Unit)** activation function is one of the most commonly used activation functions in neural networks, especially in deep learning models. It is simple, efficient, and helps mitigate some common issues in training neural networks.

### Definition

The ReLU function is mathematically defined as:
\[
f(x) = \max(0, x)
\]

This means:

- If \(x > 0\), then \(f(x) = x\).
- If \(x \leq 0\), then \(f(x) = 0\).

### Graph of RELU

The graph of the RELU function is a piecewise linear function:

- A straight line with slope 1 for positive inputs (\(x > 0\)).
- A flat horizontal line at 0 for non-positive inputs (\(x \leq 0\)).

### Advantages of RELU

1. **Computational Efficiency**:

   - ReLU is very simple to compute—it involves only a comparison and selection, making it computationally efficient compared to more complex activation functions like sigmoid or tanh.

2. **Avoids Vanishing Gradients**:

   - Unlike the sigmoid or tanh functions, which "saturate" at extreme values (causing gradients to shrink exponentially and slow down learning), ReLU maintains a gradient of 1 for positive inputs, allowing for faster learning in deep networks.

3. **Sparsity**:
   - Since ReLU outputs 0 for negative inputs, it introduces sparsity in the activations (many neurons can output 0). Sparse activations reduce model complexity and can improve computational efficiency.

### Disadvantages of RELU

1. **Dead Neurons**:

   - Some neurons can "die" during training if their weights cause them to produce outputs that are always less than or equal to 0. Once a neuron is dead (output is 0), it stops contributing to the learning process because the gradient of ReLU is 0 for negative inputs.

2. **Exploding Gradients**:
   - While ReLU prevents vanishing gradients, it doesn’t address exploding gradients, which can occur in very deep networks without proper weight initialization or regularization.

### Variants of RELU

To address its limitations, several variations of ReLU have been introduced:

1. **Leaky ReLU**:
   \[
   f(x) =
   \begin{cases}
   x & \text{if } x > 0 \\
   \alpha x & \text{if } x \leq 0
   \end{cases}
   \]
   Here, \(\alpha\) (a small positive constant like 0.01) allows a small gradient for negative inputs, reducing the risk of dead neurons.

2. **Parametric ReLU (PReLU)**:

   - Similar to Leaky ReLU, but the value of \(\alpha\) is learned during training instead of being fixed.

3. **Exponential Linear Unit (ELU)**:

   - A smooth variant that allows negative outputs but with an exponential decay.

4. **Scaled Exponential Linear Unit (SELU)**:
   - A scaled version of ELU designed for self-normalizing neural networks.

### When to Use RELU

ReLU is typically the default activation function for hidden layers in feedforward and convolutional neural networks because of its simplicity and effectiveness. However, its variants are used in specific cases where dead neurons or negative outputs are a concern.
