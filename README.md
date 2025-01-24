# transformer-from-scratch

![image](https://github.com/user-attachments/assets/eb2020e4-40ae-445b-8d1b-b894931c6b42)

References:

- https://arxiv.org/abs/1706.03762
- https://www.youtube.com/watch?v=ISNdQcPhsts

## Input Embeddings

Map words to vocabulary ids, maps vocabulary ids to 512 dimension vectors

## Positional encoding

Add a vector of the same size of the Input Embedding, that incorporates position information.

### 1. **Formula 1 (Sine for even indices):**

$`
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d\_{model}}}}\right)
`$

- **`pos`**: The position of the word in the sequence (e.g., 0 for the first word, 1 for the second word, etc.).
- **`i`**: The index of the dimension in the positional encoding vector (for even indices).
- **`d_{model}`**: The dimension of the model (e.g., 512 in many transformer models).
- **`10000^{\frac{2i}{d_{model}}}`**: A scaling factor that ensures the sine function operates on different frequencies for each dimension `i`.

The sine function is applied to all even indices (0, 2, 4, ...).

---

### 2. **Formula 2 (Cosine for odd indices):**

$`
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d\_{model}}}}\right)
`$

- Similar to the sine formula, but applied to odd indices (1, 3, 5, ...).
- The cosine function ensures that the odd-index dimensions are out of phase with the even-index dimensions.

---

### **Purpose:**

The alternating use of sine and cosine at different frequencies ensures that each position in the sequence is encoded uniquely. This helps the model differentiate between positions and encode relative distances between tokens in the sequence.

## Layer Normalization

A **normalization layer** is a component in neural networks that helps stabilize and improve training by ensuring that input or hidden layer activations have specific statistical properties. Two commonly used normalization techniques are **batch normalization** and **layer normalization**. Here’s how they work:

---

### **1. Batch Normalization**

Introduced in 2015, **batch normalization (BN)** normalizes the activations of a layer across the **batch dimension**.

#### **How It Works**:

1. **Input Statistics**:

   - For each feature (dimension) in the batch, compute the mean ($`\mu*B`$) and variance ($`\sigma_B^2`$) of the activations:
     $`
     \mu_B = \frac{1}{m} \sum*{i=1}^m x*i \quad \text{and} \quad \sigma_B^2 = \frac{1}{m} \sum*{i=1}^m (x_i - \mu_B)^2
     `$
     where $`m`$ is the batch size, and $`x_i`$ represents the activations.

2. **Normalization**:

   - Subtract the mean and divide by the standard deviation to standardize the activations:
     $`
     \hat{x}\_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
     `$
     ($`\epsilon`$ is a small constant added for numerical stability.)

3. **Scaling and Shifting**:
   - Apply a learnable scale ($`\gamma`$) and shift ($`\beta`$):
     $`
     y_i = \gamma \hat{x}\_i + \beta
     `$
   - These parameters allow the model to learn the optimal distribution of the activations.

#### **Benefits**:

- Reduces internal covariate shift (i.e., changes in layer activations during training).
- Speeds up convergence.
- Acts as a regularizer by adding noise through mini-batch statistics.

---

### **2. Layer Normalization**

Introduced later, **layer normalization (LN)** works similarly to BN but normalizes activations across the **feature dimension** rather than the batch dimension. It is especially useful for sequence models (e.g., transformers).

#### **How It Works**:

1. **Input Statistics**:

   - Compute the mean ($`\mu*L`$) and variance ($`\sigma_L^2`$) for each input sample across all features:
     $`
     \mu_L = \frac{1}{d} \sum*{j=1}^d x*j \quad \text{and} \quad \sigma_L^2 = \frac{1}{d} \sum*{j=1}^d (x_j - \mu_L)^2
     `$
     where $`d`$ is the number of features.

2. **Normalization**:

   - Normalize the activations for each feature:
     $`
     \hat{x}\_j = \frac{x_j - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}
     `$

3. **Scaling and Shifting**:
   - Like batch normalization, learnable parameters $`\gamma`$ (scale) and $`\beta`$ (shift) are applied:
     $`
     y_j = \gamma \hat{x}\_j + \beta
     `$

> gamma is also called alpha
> beta is also called bias
> We used "alpha" and "bias" in the code

#### **Benefits**:

- Works better for recurrent neural networks (RNNs) and transformers because it doesn't rely on batch size.
- Eliminates the need for large mini-batches.

---

### **Why Normalize?**

- Prevents exploding or vanishing gradients by keeping activations in a reasonable range.
- Makes the optimization landscape smoother, which helps gradient descent converge faster.
- Allows the network to train with higher learning rates, often improving performance.

---

### **When to Use Which?**

- **Batch Normalization**: Best for convolutional networks or feedforward networks with large mini-batches.
- **Layer Normalization**: Preferred for RNNs, transformers, or models with varying batch sizes (e.g., in NLP tasks).

## Feed Forward Block

A **feed forward block** in transformers is a fully connected neural network applied to each token independently. It is often referred to as a **position-wise feed forward network** because it processes each position in the sequence separately, without interaction between tokens.

#### **Mathematical Representation**:

Given an input vector $`\mathbf{X} \in \mathbb{R}^{d\_{model}}`$:

1. **Linear Transformation** (first layer):
   $`
   \mathbf{H}\_1 = \text{ReLU}(\mathbf{X} \cdot \mathbf{W}\_1 + \mathbf{b}\_1)
   `$

   - $`\mathbf{W}_1 \in \mathbb{R}^{d_{model} \times d\_{ff}}`$: Weight matrix for the first linear layer.
   - $`\mathbf{b}_1 \in \mathbb{R}^{d_{ff}}`$: Bias for the first linear layer.
   - $`d*{ff}`$: Hidden dimension size of the feed forward network (typically larger than $`d*{model}`$, e.g., $`4 \times d\_{model}`$).
   - $`\text{ReLU}`$: Non-linear activation function (other activations like GELU are sometimes used).

2. **Linear Transformation** (second layer):
   $`
   \mathbf{H}\_2 = \mathbf{H}\_1 \cdot \mathbf{W}\_2 + \mathbf{b}\_2
   `$

   - $`\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d\_{model}}`$: Weight matrix for the second linear layer.
   - $`\mathbf{b}_2 \in \mathbb{R}^{d_{model}}`$: Bias for the second linear layer.

3. **Output**:
   $`
   \mathbf{Y} = \mathbf{H}\_2
   `$

### **Benefits of the Feed Forward Block**:

1. **Non-Linearity**: The ReLU (or GELU) activation introduces non-linearities, allowing the network to learn complex mappings.
2. **Dimensional Expansion**: Expanding the dimensionality in the hidden layer enables the model to capture richer features before compressing them back.
3. **Token-Wise Processing**: Keeps the model efficient while still refining each token's representation.

---

This block is repeated in every layer of the transformer, and its parameters are learned during training.

### **Complements**

- [RELU activation function explanation](/complements/relu.md)

## Multi-Head Attention

TODO

## Residual Connection
