# LLM GPT Training from Scratch

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Size](https://img.shields.io/github/languages/code-size/simonpierreboucher02/LLM-GPT-TRAINING-FROM-0-main)](https://github.com/simonpierreboucher02/LLM-GPT-TRAINING-FROM-0-main)
[![Last Commit](https://img.shields.io/github/last-commit/simonpierreboucher02/LLM-GPT-TRAINING-FROM-0-main)](https://github.com/simonpierreboucher02/LLM-GPT-TRAINING-FROM-0-main)
[![Issues](https://img.shields.io/github/issues/simonpierreboucher02/LLM-GPT-TRAINING-FROM-0-main)](https://github.com/simonpierreboucher02/LLM-GPT-TRAINING-FROM-0-main/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/simonpierreboucher02/LLM-GPT-TRAINING-FROM-0-main)](https://github.com/simonpierreboucher02/LLM-GPT-TRAINING-FROM-0-main/pulls)

[![Transformer Architecture](https://img.shields.io/badge/Architecture-Transformer-green.svg)](https://arxiv.org/abs/1706.03762)
[![GPT Model](https://img.shields.io/badge/Model-GPT--like-orange.svg)](https://openai.com/research/language-unsupervised)
[![Multi-Head Attention](https://img.shields.io/badge/Attention-Multi--Head-purple.svg)](https://arxiv.org/abs/1706.03762)
[![From Scratch](https://img.shields.io/badge/Implementation-From%20Scratch-brightgreen.svg)](https://github.com/simonpierreboucher02/LLM-GPT-TRAINING-FROM-0-main)

## 📈 Project Status

[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)](https://github.com/simonpierreboucher02/LLM-GPT-TRAINING-FROM-0-main)
[![Maintenance](https://img.shields.io/badge/Maintenance-Well%20Maintained-green.svg)](https://github.com/simonpierreboucher02/LLM-GPT-TRAINING-FROM-0-main)
[![Documentation](https://img.shields.io/badge/Documentation-Complete-blue.svg)](https://github.com/simonpierreboucher02/LLM-GPT-TRAINING-FROM-0-main)
[![Code Quality](https://img.shields.io/badge/Code%20Quality-Production%20Ready-orange.svg)](https://github.com/simonpierreboucher02/LLM-GPT-TRAINING-FROM-0-main)

A comprehensive implementation of a GPT-like transformer model from scratch using PyTorch. This project demonstrates the complete pipeline for training and using large language models, including all core components like attention mechanisms, layer normalization, and text generation.

## 📊 Repository Metrics

| Metric | Value |
|--------|-------|
| **Files** | 5 Python files |
| **Lines of Code** | ~1,500+ lines |
| **Components** | 8 core classes |
| **Architecture** | GPT-style Transformer |
| **Implementation** | From scratch |
| **Framework** | PyTorch |

## 🚀 Features

- **Complete Transformer Implementation**: Multi-head attention, layer normalization, feed-forward networks
- **GPT Model Architecture**: Token embeddings, positional embeddings, and transformer blocks
- **Training Pipeline**: Data loading, training loops, and evaluation
- **Text Generation**: Inference and text generation capabilities
- **Modular Design**: Separated components for easy understanding and modification

## 📁 Project Structure

```
LLM-GPT-TRAINING-FROM-0-main/
├── train.py                 # Main training script with complete pipeline
├── transformer.py           # GPT model and transformer block implementations
├── Attention.py             # Multi-head attention mechanism
├── layer_normalization.py   # Layer normalization implementation
├── activation_fonction.py   # GELU activation and feed-forward networks
└── README.md               # This file
```

## 🧠 Core Components

### 1. Multi-Head Attention (`Attention.py`)
- Implements scaled dot-product attention
- Supports multiple attention heads
- Includes causal masking for autoregressive generation
- Query, Key, Value projections with optional bias

### 2. Layer Normalization (`layer_normalization.py`)
- Custom layer normalization implementation
- Learnable scale and shift parameters
- Numerical stability with epsilon parameter

### 3. Activation Functions (`activation_fonction.py`)
- GELU activation function implementation
- Feed-forward network with expansion factor of 4
- Smooth approximation of ReLU

### 4. Transformer Architecture (`transformer.py`)
- Complete GPT model implementation
- Transformer blocks with residual connections
- Token and positional embeddings
- Configurable model parameters

### 5. Training Pipeline (`train.py`)
- Data loading and preprocessing with tiktoken
- Custom dataset implementation with sliding windows
- Training and validation loops
- Text generation and evaluation
- Loss calculation and model evaluation

## 🛠️ Installation

[![pip](https://img.shields.io/badge/pip-install-blue.svg)](https://pypi.org/project/pip/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Install-red.svg)](https://pytorch.org/get-started/locally/)
[![tiktoken](https://img.shields.io/badge/tiktoken-GPT2%20Tokenizer-green.svg)](https://github.com/openai/tiktoken)
[![matplotlib](https://img.shields.io/badge/matplotlib-Plotting-orange.svg)](https://matplotlib.org/)

1. Clone the repository:
```bash
git clone https://github.com/simonpierreboucher02/LLM-GPT-TRAINING-FROM-0-main.git
cd LLM-GPT-TRAINING-FROM-0-main
```

2. Install required dependencies:
```bash
pip install torch tiktoken matplotlib numpy
```

## 📖 Usage

### Basic Text Generation

```python
from transformer import GPTModel, generate_text_simple
import tiktoken

# Model configuration
config = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# Initialize model
model = GPTModel(config)
model.eval()

# Generate text
tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

generated = generate_text_simple(model, encoded_tensor, max_new_tokens=50, context_size=1024)
generated_text = tokenizer.decode(generated[0].tolist())
print(generated_text)
```

### Training a Model

```python
from train import main

# Configuration
gpt_config = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

settings = {
    "batch_size": 4,
    "max_length": 256,
    "stride": 128,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "eval_freq": 100,
    "eval_iter": 10
}

# Start training
main(gpt_config, settings)
```

## 🔧 Model Configuration

[![Vocab Size](https://img.shields.io/badge/Vocab%20Size-50,257-blue.svg)](https://huggingface.co/gpt2)
[![Context Length](https://img.shields.io/badge/Context%20Length-1024-green.svg)](https://arxiv.org/abs/1706.03762)
[![Embedding Dim](https://img.shields.io/badge/Embedding%20Dim-768-purple.svg)](https://arxiv.org/abs/1706.03762)
[![Attention Heads](https://img.shields.io/badge/Attention%20Heads-12-orange.svg)](https://arxiv.org/abs/1706.03762)
[![Layers](https://img.shields.io/badge/Layers-12-red.svg)](https://arxiv.org/abs/1706.03762)
[![Dropout](https://img.shields.io/badge/Dropout-0.1-yellow.svg)](https://arxiv.org/abs/1706.03762)

The model supports various configurations:

- **vocab_size**: Size of the vocabulary (default: 50257 for GPT-2)
- **context_length**: Maximum sequence length (default: 1024)
- **emb_dim**: Embedding dimension (default: 768)
- **n_heads**: Number of attention heads (default: 12)
- **n_layers**: Number of transformer layers (default: 12)
- **drop_rate**: Dropout rate (default: 0.1)
- **qkv_bias**: Whether to use bias in QKV projections (default: False)

## 📊 Training Features

[![Sliding Window](https://img.shields.io/badge/Data%20Loading-Sliding%20Window-blue.svg)](https://pytorch.org/docs/stable/data.html)
[![Causal Masking](https://img.shields.io/badge/Attention-Causal%20Masking-green.svg)](https://arxiv.org/abs/1706.03762)
[![Layer Norm](https://img.shields.io/badge/Normalization-Layer%20Norm-purple.svg)](https://arxiv.org/abs/1607.06450)
[![Residual](https://img.shields.io/badge/Connections-Residual-orange.svg)](https://arxiv.org/abs/1512.03385)
[![Evaluation](https://img.shields.io/badge/Metrics-Training%20%26%20Validation-red.svg)](https://pytorch.org/docs/stable/torch.html)
[![Text Gen](https://img.shields.io/badge/Generation-Real--time-yellow.svg)](https://arxiv.org/abs/1706.03762)

- **Sliding Window Dataset**: Efficient data loading with overlapping sequences
- **Causal Masking**: Prevents attention to future tokens during training
- **Layer Normalization**: Stabilizes training with pre-norm architecture
- **Residual Connections**: Helps with gradient flow in deep networks
- **Evaluation Metrics**: Training and validation loss tracking
- **Text Generation**: Real-time text generation during training

## 🎯 Key Implementations

### Attention Mechanism
- Scaled dot-product attention
- Multi-head parallel processing
- Causal masking for autoregressive models
- Efficient tensor operations

### Transformer Block
- Self-attention with residual connections
- Feed-forward network with GELU activation
- Layer normalization for training stability
- Dropout for regularization

### Data Processing
- tiktoken tokenization (GPT-2 compatible)
- Sliding window sequence creation
- Efficient batch processing
- Memory-optimized data loading

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

[![GitHub](https://img.shields.io/badge/GitHub-simonpierreboucher02-black.svg?logo=github)](https://github.com/simonpierreboucher02)
[![Profile](https://img.shields.io/badge/Profile-View%20Profile-blue.svg)](https://github.com/simonpierreboucher02)
[![Repositories](https://img.shields.io/badge/Repos-View%20All-green.svg)](https://github.com/simonpierreboucher02?tab=repositories)

**Simon-Pierre Boucher**
- GitHub: [@simonpierreboucher02](https://github.com/simonpierreboucher02)

## 🙏 Acknowledgments

- Inspired by the original "Attention Is All You Need" paper
- Built on PyTorch framework
- Uses tiktoken for tokenization (compatible with GPT-2)

## 📚 References

- Vaswani, A., et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
- Radford, A., et al. "Language models are unsupervised multitask learners." OpenAI blog 1.8 (2019): 9.
- Brown, T., et al. "Language models are few-shot learners." Advances in neural information processing systems 33 (2020): 1877-1901.

---

⭐ If you find this project helpful, please give it a star! 