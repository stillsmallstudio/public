#!/usr/bin/env python3
import os
import nbformat as nbf

# ------------------------------------------------------------
# Notebook metadata
# ------------------------------------------------------------
notebooks = {
    "00_environment_and_overview": """# 00 — Environment Setup & Project Overview

This notebook sets up the environment for the Transformer Math Lab and outlines the structure of the entire project.

### What we'll cover
- Install and import required Python libraries
- Set random seeds and configure plotting utilities
- Project roadmap: attention → MHA → positional encodings → FFN → normalization → full transformer
- How the notebooks are organized and how to navigate them

This file acts as the starting point before diving into the math of transformers.
""",

    "01_scaled_dot_product_attention": """# 01 — Scaled Dot-Product Attention (From Scratch)

In this notebook, we build the core operation behind transformers: **scaled dot-product attention**.

### What we'll explore
- The mathematical definition of attention
- Creating queries (Q), keys (K), and values (V)
- Computing attention step-by-step (no deep learning frameworks)
- Understanding the role of the scaling factor \\( \\\\sqrt{d_k} \\)
- Visualizing attention weights

By the end, you'll fully understand the core computation powering all transformer architectures.
""",

    "02_multi_head_attention": """# 02 — Multi-Head Attention (MHA)

This notebook expands single-head attention into **multi-head attention**, allowing transformers to learn multiple interaction patterns simultaneously.

### What we'll explore
- Why multiple heads are necessary
- Splitting embeddings into heads
- Parallel attention processing
- Concatenation and output projection
- Visualizing multi-head behavior
""",

    "03_positional_encodings": """# 03 — Positional Encodings

Transformers have no inherent concept of order. This notebook shows how positional information is injected into token embeddings.

### What we'll explore
- Why positional encodings are needed
- Sinusoidal encodings
- Learned positional embeddings
- Visualizing positional patterns
""",

    "04_feed_forward_network": """# 04 — Position-Wise Feed-Forward Network (FFN)

The FFN is the nonlinear component of a transformer block and operates independently per token.

### What we'll explore
- Mathematical form of the FFN
- Hidden expansion and projection
- Activation functions (ReLU, GELU)
- Visualizing token-wise transformations
""",

    "05_layer_norm_and_residuals": """# 05 — Layer Normalization & Residual Connections

Transformers rely heavily on normalization and residual pathways for stability.

### What we'll explore
- How layer normalization works
- Pre-norm vs post-norm
- Gradient flow considerations
- Why residuals enable deep networks
""",

    "06_minimal_transformer_encoder": """# 06 — Building a Minimal Transformer Encoder

This notebook combines all components into a small, working encoder.

### What we'll explore
- Bringing MHA, FFN, norm, and residuals together
- Adding embeddings and positional encodings
- Running an end-to-end example
- Inspecting intermediate tensors
""",

    "07_interpreting_attention": """# 07 — Interpreting & Visualizing Attention

Here we inspect the patterns that emerge inside trained attention heads.

### What we'll explore
- Visual attention heatmaps
- Head specialization
- Token influence patterns
- Methods for attention probing
""",

    "08_embedding_geometry": """# 08 — Embedding Geometry & Representation Analysis

Transformer embeddings exhibit rich geometric structure.

### What we'll explore
- PCA and t-SNE visualizations
- Token similarity and cosine distances
- Clustering in embedding space
""",

    "09_training_dynamics": """# 09 — Training Dynamics of a Tiny Transformer

In this notebook we train a miniature transformer and study how its internals evolve.

### What we'll explore
- Training loop for a tiny transformer
- Evolution of attention patterns
- Loss dynamics and specialization
""",

    "10_transformer_variants_and_extensions": """# 10 — Transformer Variants & Modern Extensions

This notebook surveys modern modifications to the original transformer design.

### What we'll explore
- Decoder-only architectures (GPT)
- RoPE embeddings
- FlashAttention concepts
- Efficient attention mechanisms (Performer, Linformer)
"""
}

# ------------------------------------------------------------
# Create directory
# ------------------------------------------------------------
DIR = "transformer-math-lab"
os.makedirs(DIR, exist_ok=True)

# ------------------------------------------------------------
# Generate notebooks
# ------------------------------------------------------------
for name, markdown in notebooks.items():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(markdown))

    path = os.path.join(DIR, f"{name}.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Created: {path}")

print("\nAll notebooks created successfully!")