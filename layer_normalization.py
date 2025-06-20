# This detailed version includes explanations for the class initialization, the methods, and the parameters used within the class. It provides a clearer understanding of how the layer normalization mechanism is implemented and utilized.
# Define a class for Layer Normalization, inheriting from nn.Module
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        """
        Initialize the LayerNorm module.
        
        Parameters:
        emb_dim (int): The dimension of the embeddings.
        """
        super().__init__()
        # Small epsilon value to avoid division by zero
        self.eps = 1e-5
        # Learnable scale parameter initialized to ones
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # Learnable shift parameter initialized to zeros
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """
        Perform the forward pass of the LayerNorm.
        
        Parameters:
        x (Tensor): The input tensor of shape (batch_size, num_tokens, emb_dim).
        
        Returns:
        Tensor: The normalized output tensor of the same shape as the input.
        """
        # Calculate the mean of the input tensor along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        # Calculate the variance of the input tensor along the last dimension
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Normalize the input tensor
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # Scale and shift the normalized tensor
        return self.scale * norm_x + self.shift



