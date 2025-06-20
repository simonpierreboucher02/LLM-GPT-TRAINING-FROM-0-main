# Python code for LLM gpt like transformer training
# Simon-Pierre Boucher 
# Data loading from text file, tokeniser, transformer, decoder, inference 


# This detailed version includes explanations for each import and their purpose in the context of deep learning and data handling with PyTorch.
# Importing the tiktoken library which is used for tokenization.
import tiktoken

# Importing the torch library which is a popular deep learning framework.
import torch

# Importing the neural network module from PyTorch, which provides classes and functions to build neural networks.
import torch.nn as nn

# Importing Dataset and DataLoader from torch.utils.data. 
# Dataset is an abstract class representing a dataset, and DataLoader is used to load data from a dataset.
from torch.utils.data import Dataset, DataLoader


# This detailed version includes explanations for the class initialization, the methods, and the parameters used within the class. It provides a clearer understanding of how the dataset is created and utilized.
# Define a custom dataset class that inherits from PyTorch's Dataset class
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        Initialize the dataset with text, tokenizer, maximum length of sequences, and stride for the sliding window.
        
        Parameters:
        txt (str): The input text to be tokenized and used for creating the dataset.
        tokenizer (Tokenizer): The tokenizer to convert text to tokens.
        max_length (int): The maximum length of each sequence.
        stride (int): The stride for the sliding window.
        """
        # Initialize lists to store input and target sequences
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text and convert it into token IDs
        token_ids = tokenizer.encode(txt, allowed_special={""})

        # Use a sliding window to chunk the text into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            # Create input and target chunks using the sliding window approach
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]

            # Convert the chunks into tensors and append to the lists
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """
        Return the total number of sequences in the dataset.
        
        Returns:
        int: The number of input sequences.
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Retrieve the input and target sequence at the specified index.
        
        Parameters:
        idx (int): The index of the sequence to retrieve.
        
        Returns:
        tuple: A tuple containing the input sequence and the target sequence.
        """
        return self.input_ids[idx], self.target_ids[idx]


# This detailed version includes explanations for the function, its parameters, and the steps involved in creating the DataLoader. It provides a clearer understanding of how the DataLoader is initialized and used.

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    Create a DataLoader for the given text data using the GPTDatasetV1 class.
    
    Parameters:
    txt (str): The input text to be tokenized and used for creating the dataset.
    batch_size (int, optional): The number of samples in each batch. Default is 4.
    max_length (int, optional): The maximum length of each sequence. Default is 256.
    stride (int, optional): The stride for the sliding window. Default is 128.
    shuffle (bool, optional): Whether to shuffle the data. Default is True.
    drop_last (bool, optional): Whether to drop the last incomplete batch. Default is True.
    num_workers (int, optional): The number of worker processes for data loading. Default is 0.
    
    Returns:
    DataLoader: A DataLoader object to iterate over the dataset.
    """
    # Initialize the tokenizer with the 'gpt2' encoding
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create an instance of GPTDatasetV1 with the provided parameters
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create a DataLoader for the dataset with the specified parameters
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers
    )

    # Return the created DataLoader
    return dataloader



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

# This detailed version includes explanations for each import, providing a clearer understanding of the purpose and usage of each imported module or library in the context of the script.
# Import the os module for interacting with the operating system
import os

# Import the urllib.request module for fetching data from URLs
import urllib.request

# Import the json module for working with JSON data
import json

# Import the numpy library for numerical operations
import numpy as np

# Import the tensorflow library for building and training deep learning models
import tensorflow as tf

# Import the tqdm module for displaying progress bars
from tqdm import tqdm



def download_and_load_gpt2(model_size, models_dir):
    """
    Download and load the GPT-2 model of the specified size.
    
    Parameters:
    model_size (str): The size of the model to download. Allowed sizes are "124M", "355M", "774M", "1558M".
    models_dir (str): The directory where the model files will be stored.
    
    Returns:
    tuple: A tuple containing the model settings and parameters.
    """
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # Load settings and parameters
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination):
    """
    Download a file from a given URL to a specified destination.
    
    Parameters:
    url (str): The URL from which to download the file.
    destination (str): The path where the downloaded file will be saved.
    """
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True)

    # Get the total file size from the headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if the file already exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with the total file size
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            # Iterate over the file data in chunks
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update the progress bar
                file.write(chunk)  # Write the chunk to the file



def download_file(url, destination):
    """
    Download a file from a given URL to a specified destination.
    
    Parameters:
    url (str): The URL from which to download the file.
    destination (str): The path where the downloaded file will be saved.
    """
    # Send a GET request to download the file
    with urllib.request.urlopen(url) as response:
        # Get the total file size from the headers, defaulting to 0 if not present
        file_size = int(response.headers.get("Content-Length", 0))

        # Check if the file already exists and has the same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # Define the block size for reading the file
        block_size = 1024  # 1 Kilobyte

        # Initialize the progress bar with the total file size
        progress_bar_description = os.path.basename(url)  # Extract filename from URL
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # Open the destination file in binary write mode
            with open(destination, "wb") as file:
                # Read the file in chunks and write to the destination
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    file.write(chunk)
                    progress_bar.update(len(chunk))  # Update the progress bar




def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """
    Load GPT-2 parameters from a TensorFlow checkpoint.
    
    Parameters:
    ckpt_path (str): The path to the TensorFlow checkpoint.
    settings (dict): The settings dictionary containing model configuration.
    
    Returns:
    dict: A dictionary containing the loaded GPT-2 parameters.
    """
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params



# Importing the necessary libraries and modules

# Import the matplotlib.pyplot module for creating visualizations
import matplotlib.pyplot as plt

# Import the os module for interacting with the operating system
import os

# Import the torch library for building and training neural networks
import torch

# Import the urllib.request module for fetching data from URLs
import urllib.request

# Import the tiktoken library for tokenizing text
import tiktoken



def text_to_token_ids(text, tokenizer):
    """
    Convert text to token IDs using the specified tokenizer.
    
    Parameters:
    text (str): The input text to be tokenized.
    tokenizer (Tokenizer): The tokenizer to convert text to tokens.
    
    Returns:
    Tensor: A tensor of shape (1, seq_len) containing the token IDs.
    """
    # Encode the input text to get token IDs
    encoded = tokenizer.encode(text)
    
    # Convert the encoded tokens to a tensor and add a batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Shape: (1, seq_len)
    
    return encoded_tensor



def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs back to text using the specified tokenizer.
    
    Parameters:
    token_ids (Tensor): The input tensor of shape (1, seq_len) or (seq_len) containing the token IDs.
    tokenizer (Tokenizer): The tokenizer to convert token IDs back to text.
    
    Returns:
    str: The decoded text.
    """
    # Remove the batch dimension if it exists
    flat = token_ids.squeeze(0)  # Shape: (seq_len)
    
    # Convert the token IDs to a list and decode them to text
    return tokenizer.decode(flat.tolist())



def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate the loss for a batch of inputs and targets using the specified model.
    
    Parameters:
    input_batch (Tensor): The input tensor of shape (batch_size, seq_len) containing the input token IDs.
    target_batch (Tensor): The target tensor of shape (batch_size, seq_len) containing the target token IDs.
    model (nn.Module): The model to be used for generating logits.
    device (torch.device): The device (CPU or GPU) on which the computation will be performed.
    
    Returns:
    Tensor: The calculated loss for the batch.
    """
    # Move the input and target batches to the specified device
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    
    # Pass the input batch through the model to get the logits
    logits = model(input_batch)
    
    # Flatten the logits and target batches for the cross-entropy loss calculation
    # logits.flatten(0, 1) converts logits to shape (batch_size * seq_len, vocab_size)
    # target_batch.flatten() converts target_batch to shape (batch_size * seq_len)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    
    return loss



def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate the average loss over a specified number of batches from the data loader using the given model.
    
    Parameters:
    data_loader (DataLoader): The data loader providing batches of input and target tensors.
    model (nn.Module): The model to be used for generating logits.
    device (torch.device): The device (CPU or GPU) on which the computation will be performed.
    num_batches (int, optional): The number of batches to consider for loss calculation. 
                                 If None, the loss is calculated over all batches in the data loader.
    
    Returns:
    float: The average loss over the specified number of batches.
    """
    total_loss = 0.0
    
    # Check if the data loader is empty
    if len(data_loader) == 0:
        return float("nan")
    
    # Determine the number of batches to process
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    # Iterate over the batches in the data loader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # Calculate the loss for the current batch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()  # Accumulate the loss
        else:
            break
    
    # Calculate the average loss
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate the model on the training and validation datasets.
    
    Parameters:
    model (nn.Module): The model to be evaluated.
    train_loader (DataLoader): The data loader for the training dataset.
    val_loader (DataLoader): The data loader for the validation dataset.
    device (torch.device): The device (CPU or GPU) on which the computation will be performed.
    eval_iter (int): The number of batches to use for evaluation.
    
    Returns:
    tuple: A tuple containing the average training loss and validation loss.
    """
    # Set the model to evaluation mode to disable dropout and other training-specific layers
    model.eval()
    
    # Disable gradient calculation
    with torch.no_grad():
        # Calculate the training loss over a specified number of batches
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        
        # Calculate the validation loss over a specified number of batches
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    
    # Set the model back to training mode
    model.train()
    
    # Return the average training loss and validation loss
    return train_loss, val_loss



def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    Generate and print a text sample using the model starting from the given context.
    
    Parameters:
    model (nn.Module): The model used for generating text.
    tokenizer (Tokenizer): The tokenizer used to convert text to token IDs and vice versa.
    device (torch.device): The device (CPU or GPU) on which the computation will be performed.
    start_context (str): The initial text context to start the generation.
    """
    # Set the model to evaluation mode to disable dropout and other training-specific layers
    model.eval()
    
    # Get the context size from the model's positional embedding
    context_size = model.pos_emb.weight.shape[0]
    
    # Encode the start context to get token IDs and move them to the specified device
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    
    # Disable gradient calculation
    with torch.no_grad():
        # Generate text using the model
        token_ids = generate_text_simple(
            model=model, 
            idx=encoded,
            max_new_tokens=50, 
            context_size=context_size
        )
        
        # Decode the generated token IDs back to text
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        
        # Print the generated text in a compact format
        print(decoded_text.replace("\n", " "))
    
    # Set the model back to training mode
    model.train()




def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """
    Train the model using the specified training and validation data loaders.
    
    Parameters:
    model (nn.Module): The model to be trained.
    train_loader (DataLoader): The data loader for the training dataset.
    val_loader (DataLoader): The data loader for the validation dataset.
    optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
    device (torch.device): The device (CPU or GPU) on which the computation will be performed.
    num_epochs (int): The number of epochs to train the model.
    eval_freq (int): The frequency (in steps) to evaluate the model during training.
    eval_iter (int): The number of batches to use for evaluation.
    start_context (str): The initial text context to start the generation for sample printing.
    tokenizer (Tokenizer): The tokenizer used to convert text to token IDs and vice versa.
    
    Returns:
    tuple: A tuple containing lists of training losses, validation losses, and tokens seen.
    """
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from the previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    Plot the training and validation losses against the number of epochs and tokens seen.
    
    Parameters:
    epochs_seen (list): A list of epoch numbers corresponding to the recorded losses.
    tokens_seen (list): A list of the number of tokens seen corresponding to the recorded losses.
    train_losses (list): A list of training losses recorded at each evaluation step.
    val_losses (list): A list of validation losses recorded at each evaluation step.
    """
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs on the primary x-axis
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    # Adjust layout to make room for the labels and legend
    fig.tight_layout()
    plt.show()  # Display the plot



def main(gpt_config, settings):
    """
    Main function to train a GPT model using specified configuration and settings.
    
    Parameters:
    gpt_config (dict): Configuration dictionary for the GPT model.
    settings (dict): Settings dictionary for training parameters and hyperparameters.
    
    Returns:
    tuple: A tuple containing lists of training losses, validation losses, tokens seen, and the trained model.
    """
    # Set random seed for reproducibility
    torch.manual_seed(123)
    
    # Determine the device (CPU or GPU) to use for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # Download data if necessary
    ##############################

    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        # Download the text data from the URL
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        # Save the downloaded text data to a file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        # Read the text data from the existing file
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    ##############################
    # Initialize model
    ##############################

    # Initialize the GPT model with the specified configuration
    model = GPTModel(gpt_config)
    model.to(device)  # Move the model to the specified device

    # Initialize the optimizer with model parameters and specified settings
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation split ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    # Create the training data loader
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    # Create the validation data loader
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    ##############################
    # Train model
    ##############################

    # Initialize the tokenizer with the 'gpt2' encoding
    tokenizer = tiktoken.get_encoding("gpt2")

    # Train the model and track losses and tokens seen
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Fx Option", tokenizer=tokenizer
    )

    # Return the training and validation losses, tokens seen, and the trained model
    return train_losses, val_losses, tokens_seen, model



if __name__ == "__main__":
    """
    Main script entry point. Define configuration and settings, then run the main function.
    """
    
    # GPT model configuration for 124M parameters
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 256,  # Shortened context length (originally 1024)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }

    # Other training settings
    OTHER_SETTINGS = {
        "learning_rate": 5e-4,  # Learning rate for the optimizer
        "num_epochs": 10,       # Number of epochs to train the model
        "batch_size": 2,        # Batch size for training and validation
        "weight_decay": 0.1     # Weight decay for the optimizer
    }

    # Call the main function with the specified configuration and settings
    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)


