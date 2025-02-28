"""
* Implementing a GPT model from scratch to generate text.
- Coding a GPM-like LLM that can be trained to generate human-like text
- Normalizeing layer activations to stablize neural network training.
- Adding shortcut connections in deep newural networks to train models more effectively.
- Computing the number of parameters and storage requirements of GPT models.

"""

GPT_CONFIG_124M = {
    "vocab_size": 50257,  #BPE tokenizer vocabulary of 50,257 words
    "context_length": 1024,  #maximum number of input tokens the model can handle.
    "emb_dim": 768, #embedding size, transforming each token into a 768-dimension vector.
    "n_heads": 12,  # count of attention heads in the multi-head attention mechanism
    "n_layers": 12, # number of transformer blocks in the model
    "drop_rate":0.1,    # intensity of the dropout ( i.e., 1-% drop of hidden units) to prevent overfitting.
    "qkv_bias": False   # whether to include a bias vector in the Linear layers of the multi-head agttention for query, key, and value computations. 
}

import torch
import torch.nn as nn

#*Coding an LLM architecture

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #Defines a GPT-like model 
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias = False)

    def forward(self, in_idx):
        #describe the data flow through the model
        batch_size, seq_len = in_idx.shape
        #computes tokens and positional embeddings for input indices
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds =  self.pos_emb(torch.arange(seq_len, device = in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x) #apply dropouts
        x = self.trf_blocks(x) #process the data through transformer blocks
        x = self.final_norm(x) # apply normalization
        logits = self.out_head(x) #porudces logits with the linear output layer.
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    
    def forward(self, x):
        return x

# The token embedding is handled inside the GPT model. 
# In LLM, the embedded input token dimension typically matches the output dimension

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

#Tokenize a batch consisting two text inputs
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
# print(batch)
# tensor([[6109, 3626, 6100,  345],
#         [6109, 1110, 6622,  257]])

# Initialize a new 124 million parameter DummyGPTModel instance and feed it with the tokenized batch

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
# print("Output shape: ", logits.shape)
# print(logits)

# Output shape:  torch.Size([2, 4, 50257]) #two rows--> two text samples. Each consists 4 tokens, each token is a 50257-dimensional vector--> matches the size of tokenizer's vocabulary
# tensor([[[-0.9289,  0.2748, -0.7557,  ..., -1.6070,  0.2702, -0.5888],
#          [-0.4476,  0.1726,  0.5354,  ..., -0.3932,  1.5285,  0.8557],
#          [ 0.5680,  1.6053, -0.2155,  ...,  1.1624,  0.1380,  0.7425],
#          [ 0.0448,  2.4787, -0.8843,  ...,  1.3219, -0.0864, -0.5856]],

#         [[-1.5474, -0.0542, -1.0571,  ..., -1.8061, -0.4494, -0.6747],
#          [-0.8422,  0.8243, -0.1098,  ..., -0.1434,  0.2079,  1.2046],
#          [ 0.1355,  1.1858, -0.1453,  ...,  0.0869, -0.1590,  0.1552],
#          [ 0.1666, -0.8138,  0.2307,  ...,  2.5035, -0.3055, -0.3083]]],
#        grad_fn=<UnsafeViewBackward0>)