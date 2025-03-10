import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model, vocab_size: int):
        super().__init__()
        self.d_model = d_model # this tells us the dimension of the model, mapping between the numbers and vectors of size 512  
        self.vocab-size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # dictionary of numbers to vectors 

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model : int, seq_len : int, dropout : float ) -> None:
        super().__init__()
        self.d_model = d_model # no of dimensions
        self.seq_len = seq_len # length of the sequence/ sentence
        self.dropout = nn.Dropout(dropout) # prevents overfit

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) #
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[: , 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[: , :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied, learnable parameter by using nn
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) # usually this takes  off the dim( here x) but we want it
        std = x.std( dim = -1, keepdim = True)
        return self.alpha * ( x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout : float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) #w1 & b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #w2 & b2

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    

