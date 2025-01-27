import torch
import torch.nn as nn
import math

from attention import attention

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        d_model: dimension of the embedding vector
        vocab_size: size of the vocabulary (number of unique tokens)
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, x):
        """
        x: input vector of shape (batch_size, seq_len)
        """
        # Multiply the embedding vector by the square root of the dimension of the embedding vector to scale the embeddings
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        d_model: dimension of the embedding vector
        seq_len: length of the input sequence
        dropout: dropout rate, used to prevent overfitting
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model) containing the positional encodings
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of seq_len dimensions containing the positional encodings
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Expression from formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a new dimension to the positional encodings in the first index to match the batch size -- (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register the positional encodings as a buffer
        # Keep the positional encodings as part of the model state
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: input embedding vector
        """
        # Retrieve the positional encodings corresponding to the current input sequence length
        # "requires_grad_(False)" is used to prevent PyTorch from computing gradients for the positional encodings since they are fixed
        positional_encodings = (self.pe[:, :x.size(1), :]).requires_grad_(False)
        # Add the input embeddings to the positional encodings to incorporate positional information
        combined = x + positional_encodings
        # Dropout is a regularization technique used to prevent overfitting by randomly setting a fraction of the input units to zero
        return self.dropout(combined)
    

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        """
        d_model: dimension of the embedding vector
        eps: small value to prevent division by zero
        """
        super().__init__()
        self.eps = eps

        # Learnable parameters
        self.alpha = nn.Parameter(torch.ones(d_model)) # multiplied
        self.bias = nn.Parameter(torch.zeros(d_model)) # Added

    def forward(self, x):
        """
        x: input batch of embedding vectors of shape (batch_size, seq_len, d_model)
        """
        # Calculate the mean and variance of the input vector
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # Normalize the input vector
        norm = (x - mean) / (std + self.eps) 
        return self.alpha * norm + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        d_model: dimension of the embedding vector
        d_ff: dimension of the hidden layer of the feedforward network
        dropout: dropout rate, used to prevent overfitting
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # nn.Linear applies a linear transformation to the incoming data: y = xW^T + b
        # x is the input data, W is the weight matrix, and b is the bias
        # it is also known as a fully connected layer or dense layer

        # Creates a linear layer that transforms the input vector to a higher-dimensional space (d_ff)
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        # Creates a linear layer that transforms the input vector back to the original dimension (d_model)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        """
        x: input batch of embedding vectors of shape (batch_size, seq_len, d_model)

        (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        """
        # Apply the first linear transformation
        x = self.linear_1(x)
        # Apply the ReLU activation function
        x = torch.relu(x)
        # Apply dropout
        x = self.dropout(x)
        # Apply the second linear transformation
        x = self.linear_2(x)
        return x
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """
        d_model: dimension of the embedding vector
        h: number of heads
        dropout: dropout rate, used to prevent overfitting
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"

        # d_k is the dimension of the query, key and value vectors in each head
        self.d_k = d_model // h

        # Query, key, and value weight matrices
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)  

        # Output weight matrix
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):
        """
        q: query vector of shape (batch_size, seq_len, d_model)
        k: key vector of shape (batch_size, seq_len, d_model)
        v: value vector of shape (batch_size, seq_len, d_model)
        mask: mask to prevent attention to certain elements
        """
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        # Basically, we are multiplying the input query, key, and value vectors by the weight matrices
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split the query, key, and value vectors into h heads (smaller matrices)
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -transpose-> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Apply the attention mechanism to the query, key, and value vectors
        x, self.attention_scores = attention(query, key, value, self.dropout, mask)

        # (batch_size, h, seq_len, d_k) -transpose-> (batch_size, seq_len, h, d_k)
        # Combine the h heads into a single d_model-sized vector
        # (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        # Return the value multiplied by the output weight matrix
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        """
        dropout: dropout rate, used to prevent overfitting
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        """
        x: input batch of embedding vectors of shape (batch_size, seq_len, d_model)
        sublayer: output of the previous layer (MultiHeadAttentionBlock or FeedForward)

        It performs the skip connection by adding the input vector to the output of the previous layer (sublayer)
        """
        # Layer normalize the input vector and multiply it by the previous layer output vector
        # Then apply dropout and add the input vector
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        """
        x: input batch of embedding vectors of shape (batch_size, seq_len, d_model)
        mask: mask to prevent attention to certain elements
        """
        # Performs the first skip connection (to the multi-head-attention block)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        # Performs the second skip connection (to the feedforward block)
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        """
        layers: list of EncoderBlock layers
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask):
        """
        x: input batch of embedding vectors of shape (batch_size, seq_len, d_model)
        mask: mask to prevent attention to certain elements
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.source_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        x: input batch of embedding vectors of shape (batch_size, seq_len, d_model)
        encoder_output: output of the encoder (batch_size, seq_len, d_model)
        src_mask: mask to the tokens coming from the encoder
        tgt_mask: mask to the tokens coming from the decoder
        """
        # Performs the first skip connection (to the self-attention block)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # Performs the second skip connection (to the source-attention block)
        x = self.residual_connections[1](x, lambda x: self.source_attention_block(x, encoder_output, encoder_output, src_mask))
        # Performs the third skip connection (to the feedforward block)
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model:int, layers: nn.ModuleList) -> None:
        """
        layers: list of DecoderBlock layers
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        x: input batch of embedding vectors of shape (batch_size, seq_len, d_model)
        encoder_output: output of the encoder (batch_size, seq_len, d_model)
        src_mask: mask to the tokens coming from the encoder
        tgt_mask: mask to the tokens coming from the decoder
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        d_model: dimension of the embedding vector
        vocab_size: size of the vocabulary (number of unique tokens)
        """
        super().__init__()
        # Creates a linear layer to project the output of the decoder to the vocabulary size
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: input batch of embedding vectors of shape (batch_size, seq_len, d_model)
        """
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.linear(x)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        src: input batch of source tokens
        src_mask: mask to prevent attention to certain source tokens
        """
        # Embed the source tokens and apply positional encoding
        # Then pass the embeddings through the encoder
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """
        encoder_output: output of the encoder
        src_mask: mask to the tokens coming from the encoder
        tgt: input batch of target tokens
        tgt_mask: mask to the tokens coming from the decoder
        """
        # Embed the target tokens and apply positional encoding
        # Then pass the embeddings through the decoder
        return self.decoder(self.tgt_pos(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        """
        x: decoder output of shape (batch_size, seq_len, d_model)
        """
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, h: int = 8, N: int = 6,  d_ff = 2048, dropout: float = 0.1) -> Transformer:
    """
    Creates a Transformer instance optimized for machine translation tasks
    """
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # TODO: optimize it to use only one positional encoding
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)    

    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(d_model, self_attention_block, feed_forward_block, dropout))

    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(d_model, self_attention_block, cross_attention_block, feed_forward_block, dropout))

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters with xavier uniform initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
