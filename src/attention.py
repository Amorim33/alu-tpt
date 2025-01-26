import torch
import torch.nn as nn
import math

def attention(query, key, value, dropout: nn.Dropout, mask=None):
    """
    query: query vector of shape (batch_size, h, seq_len, d_k)
    key: key vector of shape (batch_size, h, seq_len, d_k)
    value: value vector of shape (batch_size, h, seq_len, d_k)
    mask: mask to prevent attention to certain elements
    """
    d_k = query.shape[-1]

    # Calculate the dot product of the query and key vectors
    # (batch_size, h, seq_len, d_k) x (batch_size, h, d_k, seq_len) -> (batch_size, h, seq_len, seq_len)
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply the mask to prevent attention to certain elements
    if mask is not None:
        # This operation will set the attention scores to -10⁹ for all masked elements
        attention_scores = attention_scores.masked_fill_(mask == 0, -1e4)

    # Apply the softmax function to normalize the attention_scores
    # (batch_size, h, seq_len, seq_len)
    attention_scores = attention_scores.softmax(dim=-1)

    # Apply dropout
    attention_scores = dropout(attention_scores)

    # Calculate the weighted sum of the value vectors
    # (batch_size, h, seq_len, seq_len) x (batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, d_k)
    return torch.matmul(attention_scores, value), attention_scores
