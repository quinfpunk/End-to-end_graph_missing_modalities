import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# segmenting the time series into windows
def segment_time_series(data, window_size):
    """
    Groups time series data into windows.
    
    Args:
        data (list): List of tuples (timestamp, [(test_name, test_value), ...])
        window_size (int): Time window size (e.g., in time units)
    
    Returns:
        list: A list of windows. Each window is a list of (timestamp, tests) tuples
    """
    windows = []
    current_window = []
    start_time = data[0][0]

    for timestamp, tests in data:
        if timestamp - start_time < window_size:
            current_window.append((timestamp, tests))
        else:
            windows.append(current_window)
            current_window = [(timestamp, tests)]
            start_time = timestamp
    if current_window:
        windows.append(current_window)
    return windows

# window encoder: embedding a single window
class WindowEncoder(nn.Module):
    """
    Encodes a single time window into a fixed-size embedding
    It embeds the test names and processes test values,
    then aggregates the test-level representations (mean pooling)
    """
    def __init__(self, vocab_size, name_embed_dim, value_dim, hidden_dim):
        super(WindowEncoder, self).__init__()
        if name_embed_dim != 0:
            self.name_embedding = nn.Embedding(vocab_size, name_embed_dim)
        self.value_embedding = nn.Linear(1, value_dim)
        self.fc = nn.Linear(name_embed_dim + value_dim, hidden_dim)
        self.activation = nn.ReLU()
    
    def forward(self, test_names, test_values):
        """
        Args:
            test_names (Tensor): Tensor of test name indices of shape (num_tests,)
            test_values (Tensor): Tensor of test values of shape (num_tests,)
        
        Returns:
            Tensor: A window embedding of shape (hidden_dim,)
        """
        # Embed test names
        if test_names.numel() == 0:
            name_embeds = torch.empty(0)
        else:
            name_embeds = self.name_embedding(test_names)  # shape (num_tests, name_embed_dim)
        # Process test values (unsqueeze to shape (num_tests, 1))
        test_values = test_values.unsqueeze(1)
        value_embeds = self.value_embedding(test_values)  # shape (num_tests, value_dim)
        
        # Combine the two embeddings
        combined = torch.cat([name_embeds, value_embeds], dim=-1)  # (num_tests, name_embed_dim + value_dim)
        combined = self.activation(self.fc(combined))  # shape (num_tests, hidden_dim)
        
        # Aggregate test-level embeddings (mean pooling)
        window_embedding = combined.mean(dim=0)  # (hidden_dim,)
        return window_embedding

# TimeSeriesBERT: the overall model
class TimeSeriesBERTCLS(nn.Module):
    def __init__(self, vocab_size, name_embed_dim, value_dim, hidden_dim, 
                 num_layers, num_heads, max_seq_len, dropout=0.1):
        super(TimeSeriesBERTCLS, self).__init__()
        self.max_seq_len = max_seq_len
        self.window_encoder = WindowEncoder(vocab_size, name_embed_dim, value_dim, hidden_dim)
        
        # define a learnable CLS token embedding
        self.cls_token = nn.Parameter(torch.randn(1, hidden_dim))
        self.positional_embedding = nn.Embedding(max_seq_len + 1, hidden_dim)  # +1 for CLS token
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        # downstream head: outputs fixed shape (hidden_dim,)
        self.head = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, windows_batch):
        """
        windows_batch: list of window tokens. We'll prepend the CLS token.
        """
        window_embeddings = []
        for test_names, test_values in windows_batch:
            emb = self.window_encoder(test_names, test_values)
            window_embeddings.append(emb)
        
        # optionally pad/truncate window_embeddings to self.max_seq_len
        seq_len = len(window_embeddings)
        if seq_len < self.max_seq_len:
            pad_tensor = torch.zeros(self.max_seq_len - seq_len, window_embeddings[0].shape[-1],
                                     device=window_embeddings[0].device)
            window_embeddings = window_embeddings + [pad_tensor[i] for i in range(pad_tensor.size(0))]
        else:
            window_embeddings = window_embeddings[:self.max_seq_len]
        
        # stack and prepend the CLS token: shape becomes (max_seq_len+1, hidden_dim)
        window_embeddings = torch.stack(window_embeddings, dim=0)
        cls_token = self.cls_token  # shape (1, hidden_dim)
        x = torch.cat([cls_token, window_embeddings], dim=0)
        
        # add positional embeddings (we now have max_seq_len+1 positions)
        positions = torch.arange(x.size(0), device=x.device)
        pos_embeds = self.positional_embedding(positions)
        x = x + pos_embeds
        x = self.dropout(x)
        
        # transformer expects (seq_len, batch_size, hidden_dim)
        x = x.unsqueeze(1)  # shape (max_seq_len+1, 1, hidden_dim)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)  # shape (max_seq_len+1, hidden_dim)
        
        # use only the CLS token's output as the fixed-size representation
        cls_output = x[0]  # shape (hidden_dim,)
        output = self.head(cls_output)
        return output
