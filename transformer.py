# This detailed version includes explanations for the class initialization, the forward method, and the components of the transformer block. It provides a clearer understanding of how the transformer block is structured, incorporating multi-head attention, layer normalization, feed-forward networks, and residual connections.
# Define a class for a Transformer Block, inheriting from nn.Module
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        """
        Initialize the TransformerBlock module.
        
        Parameters:
        cfg (dict): Configuration dictionary containing the necessary parameters.
                    Example:
                    {
                        "emb_dim": 128,
                        "context_length": 512,
                        "n_heads": 8,
                        "drop_rate": 0.1,
                        "qkv_bias": True
                    }
        """
        super().__init__()
        # Initialize the MultiHeadAttention layer
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        # Initialize the FeedForward network
        self.ff = FeedForward(cfg)
        # Initialize the Layer Normalization layers
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # Initialize the Dropout layer for the shortcut connections
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        Perform the forward pass of the Transformer Block.
        
        Parameters:
        x (Tensor): The input tensor of shape (batch_size, num_tokens, emb_dim).
        
        Returns:
        Tensor: The output tensor of the same shape as the input.
        """
        # Shortcut connection for the attention block
        shortcut = x
        x = self.norm1(x)           # Apply layer normalization
        x = self.att(x)             # Apply multi-head attention
        x = self.drop_shortcut(x)   # Apply dropout
        x = x + shortcut            # Add the original input back (residual connection)

        # Shortcut connection for the feed-forward block
        shortcut = x
        x = self.norm2(x)           # Apply layer normalization
        x = self.ff(x)              # Apply feed-forward network
        x = self.drop_shortcut(x)   # Apply dropout
        x = x + shortcut            # Add the original input back (residual connection)

        return x

# This detailed version includes explanations for the class initialization, the forward method, and the components of the GPT model. It provides a clearer understanding of how the GPT model is structured, including token and positional embeddings, stacked transformer blocks, final normalization, and output layer.
# Define a class for a GPT model, inheriting from nn.Module
class GPTModel(nn.Module):
    def __init__(self, cfg):
        """
        Initialize the GPTModel module.
        
        Parameters:
        cfg (dict): Configuration dictionary containing the necessary parameters.
                    Example:
                    {
                        "vocab_size": 50257,
                        "emb_dim": 768,
                        "context_length": 1024,
                        "drop_rate": 0.1,
                        "n_layers": 12,
                        "n_heads": 12,
                        "qkv_bias": True
                    }
        """
        super().__init__()
        # Token embedding layer
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # Positional embedding layer
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # Dropout layer for embeddings
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Stacked Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final layer normalization
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # Output linear layer (vocabulary size output)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        Perform the forward pass of the GPT model.
        
        Parameters:
        in_idx (Tensor): The input tensor of shape (batch_size, seq_len).
        
        Returns:
        Tensor: The logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        # Get the batch size and sequence length from the input tensor
        batch_size, seq_len = in_idx.shape
        # Get token embeddings
        tok_embeds = self.tok_emb(in_idx)
        # Get positional embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # Add token and positional embeddings
        x = tok_embeds + pos_embeds  # Shape [batch_size, seq_len, emb_dim]
        # Apply dropout to the embeddings
        x = self.drop_emb(x)
        # Pass through the stacked Transformer blocks
        x = self.trf_blocks(x)
        # Apply final layer normalization
        x = self.final_norm(x)
        # Get logits from the output linear layer
        logits = self.out_head(x)
        
        return logits


# This detailed version includes explanations for the function, its parameters, and the steps involved in generating text. It provides a clearer understanding of how the text generation process works using a language model, including context cropping, prediction, and appending the predicted tokens to the running sequence.
def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text using the given model by predicting the next token iteratively.
    
    Parameters:
    model (nn.Module): The language model used for generating text.
    idx (Tensor): The input tensor of shape (batch_size, seq_len) representing the current context.
    max_new_tokens (int): The maximum number of new tokens to generate.
    context_size (int): The size of the context window supported by the model.
    
    Returns:
    Tensor: The tensor of shape (batch_size, seq_len + max_new_tokens) representing the generated text.
    """
    # Iterate to generate new tokens
    for _ in range(max_new_tokens):
        # Crop the current context if it exceeds the supported context size
        # For example, if the model supports only 5 tokens as context and the context size is 10,
        # only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions from the model
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # The shape (batch_size, seq_len, vocab_size) becomes (batch_size, vocab_size)
        logits = logits[:, -1, :]

        # Get the index of the vocabulary entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # Shape: (batch_size, 1)

        # Append the sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # Shape: (batch_size, seq_len + 1)

    return idx


# This detailed version includes explanations for the main block of the script, covering the configuration setup, model initialization, tokenizer usage, text encoding, and the text generation process. It provides a clearer understanding of how the script sets up the GPT model, processes the input text, generates new text, and decodes the output.
if __name__ == "__main__":
    # Configuration dictionary for the GPT model
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    # Set the random seed for reproducibility
    torch.manual_seed(123)
    # Initialize the GPT model with the given configuration
    model = GPTModel(GPT_CONFIG_124M)
    # Set the model to evaluation mode to disable dropout
    model.eval()

    # Starting context for text generation
    start_context = "Hello, I am"

    # Initialize the tokenizer with the 'gpt2' encoding
    tokenizer = tiktoken.get_encoding("gpt2")
    # Encode the starting context to get token IDs
    encoded = tokenizer.encode(start_context)
    # Convert the encoded tokens to a tensor and add a batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    # Print input information
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    # Generate new text using the model
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    # Decode the generated token IDs back to text
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    # Print output information
    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)
