### **00 — Environment Setup & Project Overview**
- Sets up device (MPS for M1/M2 Macs or CPU)
- Configures reproducible random seeds
- Defines utility functions for visualizations
- Documents the roadmap for the full project

### **01 — Scaled Dot-Product Attention**
- Construct Q, K, and V matrices by hand
- Compute attention: dot-product → scale → softmax → weighted sum
- Visualize attention matrices
- Build intuition for token interactions

### **02 — Multi-Head Attention**
- Split embeddings into multiple heads
- Perform parallel attention computations
- Concatenate outputs and apply projections
- Show how different heads learn different behaviors

### **03 — Positional Encodings**
- Explain why transformers need positional information
- Implement sinusoidal encodings from the original paper
- Compare with learned positional embeddings
- Visualize frequency patterns

### **04 — Feed-Forward Networks (FFN)**
- Implement the position-wise two-layer MLP used in each block
- Explore activation functions (ReLU, GELU)
- Understand how FFNs transform token embeddings independently

### **05 — Layer Normalization & Residual Connections**
- Compare pre-norm vs post-norm transformers
- Show how residuals stabilize deep networks
- Visualize gradient flow and normalization effects

### **06 — Minimal Transformer Encoder**
- Combine embeddings, positional encodings, MHA, LN, FFN, and residual pathways
- Build a fully functional encoder layer from scratch
- Run simple sequences through it
- Inspect every intermediate tensor

### **07 — Interpreting Attention**
- Visualize attention heatmaps
- Explore how heads specialize
- Analyze token-to-token influence patterns
- Identify positional, syntactic, and semantic heads

### **08 — Embedding Geometry**
- Project embeddings into 2D (PCA/t-SNE)
- Look at clusters and directions in embedding space
- Compute cosine similarities
- Explore anisotropy and structure

### **09 — Training Dynamics**
- Train a tiny transformer encoder
- Track how attention evolves over training
- Analyze loss curves and head specialization
- Visualize before/after transformations

### **10 — Transformer Variants & Extensions**
- GPT-style decoder-only architecture
- Rotary Positional Embeddings (RoPE)
- FlashAttention concepts
- Efficient attention mechanisms (Performer, Linformer, etc.)
- Connect foundational concepts to modern large models

