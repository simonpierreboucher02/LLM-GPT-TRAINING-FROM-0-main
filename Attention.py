# This detailed version includes explanations for the class initialization, the methods, and the parameters used within the class. It provides a clearer understanding of how the multi-head attention mechanism is implemented and utilized.
# Define a class for Multi-Head Attention, inheriting from nn.Module
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        Initialize the Multi-Head Attention module.
        
        Parameters:
        d_in (int): The dimension of the input.
        d_out (int): The dimension of the output.
        context_length (int): The length of the context window.
        dropout (float): Dropout rate.
        num_heads (int): The number of attention heads.
        qkv_bias (bool, optional): Whether to add bias to query, key, and value projections. Default is False.
        """
        super().__init__()
        # Ensure that the output dimension is divisible by the number of heads
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Dimension of each head

        # Linear layers for query, key, and value projections
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Register a buffer for the causal mask to prevent attending to future tokens
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        Perform the forward pass of the Multi-Head Attention.
        
        Parameters:
        x (Tensor): The input tensor of shape (batch_size, num_tokens, d_in).
        
        Returns:
        Tensor: The output tensor of shape (batch_size, num_tokens, d_out).
        """
        # Get the shape of the input tensor
        b, num_tokens, d_in = x.shape

        # Compute the query, key, and value projections
        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape to add the num_heads dimension and split the d_out dimension into num_heads and head_dim
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose to bring the num_heads dimension before the num_tokens dimension
        keys = keys.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Apply the causal mask to the attention scores
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Compute the attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute the context vectors by applying the attention weights to the values
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Reshape to combine the heads
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        
        # Apply the output projection
        context_vec = self.out_proj(context_vec)

        return context_vec
