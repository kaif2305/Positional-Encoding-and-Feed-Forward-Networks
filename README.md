# Positional Encoding and Transformer Block Implementation

This project explores the concept of **Positional Encoding**, a crucial component of the Transformer architecture, and demonstrates how it's integrated into a simplified Transformer block using PyTorch. It covers both the original sinusoidal positional encoding and a conceptual learnable positional encoding.

## What is Positional Encoding?

In traditional Recurrent Neural Networks (RNNs), the sequential nature of data is inherently captured by processing tokens one after another. However, in Transformer networks, multi-head attention processes all tokens in a sequence simultaneously, losing information about their relative or absolute positions.

**Positional Encoding** addresses this by injecting information about the position of each token into its embedding. These positional encodings are added to the word embeddings before they are fed into the Transformer's encoder and decoder stacks. This allows the self-attention mechanism to understand the order of words in a sequence.

There are primarily two types:

1.  **Sinusoidal Positional Encoding (Fixed/Non-Learnable)**: As proposed in the original "Attention Is All You Need" paper, this uses sine and cosine functions of different frequencies. It's fixed and not learned during training, making it generalizable to unseen sequence lengths.
    * For even indices $i$: $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$
    * For odd indices $i$: $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$
    Where $pos$ is the position and $d_{\text{model}}$ is the embedding dimension.

2.  **Learnable Positional Encoding**: In some variations, positional embeddings are randomly initialized and then learned directly during the model's training process. This offers flexibility but might not generalize as well to sequence lengths much longer than those seen during training.

## Project Overview

This Python script demonstrates:

1.  **Sinusoidal Positional Encoding Generation**: A NumPy function generates fixed positional encodings.
2.  **Visualization of Positional Encoding**: A heatmap visualizes the pattern of the generated sinusoidal positional encodings.
3.  **Transformer Block with Fixed Positional Encoding**: A PyTorch `nn.Module` class (`TransformerWithPositionalEncoding`) implements a simplified Transformer block that adds the pre-computed sinusoidal positional encodings to word embeddings.
4.  **Learnable Positional Encoding**: A conceptual PyTorch `nn.Module` class (`LearnablePositionalEncoding`) shows how learnable positional embeddings would be structured.

## Part 1: Sinusoidal Positional Encoding (NumPy & Visualization)

The `positional_encoding` function calculates the sine and cosine values for various positions and embedding dimensions, creating the unique "position signature."

```python
import numpy as np
import matplotlib.pyplot as plt

# Define positional encoding function
def positional_encoding(seq_len, embed_dim):
    pos = np.arange(seq_len)[:, np.newaxis] # Positions (0, 1, ..., seq_len-1)
    i = np.arange(embed_dim)[np.newaxis, :] # Dimensions (0, 1, ..., embed_dim-1)
    # Calculate division term for angles
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
    angle_rads = pos * angle_rates
    
    # Apply sine to even indices and cosine to odd indices
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2]) # Even indices use sine
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2]) # Odd indices use cosine
    return pos_encoding

# Generate and visualize positional encoding
seq_len = 50      # Example sequence length
embed_dim = 16    # Example embedding dimension
pos_encoding = positional_encoding(seq_len, embed_dim)

# Visualize positional encoding as a heatmap
plt.figure(figsize=(10,6))
plt.pcolormesh(pos_encoding, cmap='viridis')
plt.colorbar()
plt.title("Positional Encoding")
plt.xlabel("Embedding Dimension")
plt.ylabel("Position")
plt.show()