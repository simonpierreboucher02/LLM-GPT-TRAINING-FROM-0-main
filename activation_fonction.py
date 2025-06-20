# This detailed version includes explanations for the class initialization and the forward method, providing a clearer understanding of how the GELU activation function is implemented and utilized. The GELU (Gaussian Error Linear Unit) activation function is a smooth approximation of the ReLU activation function and is commonly used in modern neural network architectures.
# Define a class for GELU activation function, inheriting from nn.Module
class GELU(nn.Module):
    def __init__(self):
        """
        Initialize the GELU module.
        """
        super().__init__()

    def forward(self, x):
        """
        Perform the forward pass of the GELU activation function.
        
        Parameters:
        x (Tensor): The input tensor.
        
        Returns:
        Tensor: The output tensor after applying the GELU activation function.
        """
        # Compute the GELU activation using the approximation formula
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))



# This detailed version includes explanations for the class initialization, the forward method, and the structure of the feedforward neural network. It provides a clearer understanding of how the feedforward network is implemented and utilized within a larger neural network architecture. The feedforward network typically includes linear transformations and activation functions, with the given example using the GELU activation function and two linear layers.
# Define a class for FeedForward neural network, inheriting from nn.Module
class FeedForward(nn.Module):
    def __init__(self, cfg):
        """
        Initialize the FeedForward module.
        
        Parameters:
        cfg (dict): Configuration dictionary containing the embedding dimension.
                    Example: {"emb_dim": 128}
        """
        super().__init__()
        # Define a sequential container with the feedforward network layers
        self.layers = nn.Sequential(
            # First linear layer: projects from emb_dim to 4 * emb_dim
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # GELU activation function
            GELU(),
            # Second linear layer: projects from 4 * emb_dim back to emb_dim
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """
        Perform the forward pass of the FeedForward network.
        
        Parameters:
        x (Tensor): The input tensor of shape (batch_size, num_tokens, emb_dim).
        
        Returns:
        Tensor: The output tensor of the same shape as the input.
        """
        # Pass the input through the sequential layers
        return self.layers(x)


