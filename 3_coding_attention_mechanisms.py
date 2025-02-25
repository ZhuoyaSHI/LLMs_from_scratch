"""
    *A simple self-attention mechanism without trainable weights.

    When computing the context vector z(2). The inportance/contribution of each input element for computing z(2) is determined
    by the attention weights alpha(21) to alpha(2T). When computing z(2), the attention weights are calculated with respect to 
    input element  x(2) and all other input elements.
"""

import torch

inputs =  torch.tensor(
    [
        [0.43, 0.15, 0.89], # Your (x^1)
        [0.55, 0.87, 0.66], # journey (x^2)
        [0.57, 0.85, 0.64], # starts (x^3)
        [0.22, 0.58, 0.33], # with (x^4)
        [0.77, 0.25, 0.10], # one (x^5)
        [0.05, 0.80, 0.55]  # step (x^6)
    ]
)

"""
*The first intermediate step: Compute the attention score w between the query x^2 and all other input elements as a dot product.

    dot product: multiply two vectors elementwise and then sum the products.
                 A methematical tool that combines two vectors to yield a scalar value.
                 A measure of similarity because it quantifies how much two vectors are aligned. 
                 (higher --> greater degree of alignement/similarity)
"""

query = inputs[1] 
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot( x_i, query)
    
# print(attn_scores_2) # tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

"""
*Step 2: Obtain the attention weights alpha21 to alpha2T by normalizing the attention scores.
    The normalization process ensures that all attention weights sum up to 1.
"""
attn_weights_2_tmp = attn_scores_2/attn_scores_2.sum()
# print("Attention weights: ", attn_weights_2_tmp)  #Attention weights:  tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
# print("Sum: ", attn_weights_2_tmp.sum())  #Sum:  tensor(1.0000)



# In Practice, Use softmax function for normalization.
# The softmax function ensures that the attention weights are always positive.
# --> Output interpretavle as probabilities or relative importance. (higher--> greater importance).


def softmax_naive(x):
    return torch.exp(x)/torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

# print("Attention weights: ", attn_weights_2_naive)
# print("Sum: ", attn_weights_2_naive.sum())

#Note: Naive softmax implementation may encounter numerical instability 
#(e.g., overflow & underflow) when dealing with large or small input values.
#--> PyTorch implementation of softmax

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# print("Attention weights: ", attn_weights_2)
# print("Sum: ", attn_weights_2.sum())

"""
*Step 3: Calculate the context vector z^2 by multiplying the embedded input tokens x^i with the corresponding attention weights 
and sum the resulting vector.
"""

query = inputs[1] #2nd input token 
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

# print("Context vector: ", context_vec_2) #Context vector:  tensor([0.4419, 0.6515, 0.5683])

"""
*Implementation on the whole sentence input.
"""
# Step 1: 
# Compute attention weights for all input tokens

attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
# print(attn_scores)

# tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
#         [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
#         [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
#         [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
#         [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
#         [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])

#Each element represents an attention score between each pair of inputs.

#Tips:
#Instead of for-loops, using matrix multiplication to achieve the same result.
attn_scores = inputs @ inputs.T
# print(attn_scores)

#Step 2: 
# normalize each row
attn_weights = torch.softmax(attn_scores, dim=1)
# print(attn_weights)

# tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
#         [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
#         [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
#         [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
#         [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
#         [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])

#Verify the rows all sum to 1

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# print("Row 2 sum: ", row_2_sum) #Row 2 sum:  1.0
# print("All row sums: ", attn_weights.sum(dim=1)) #All row sums:  tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

#Step 3:
#Computer all context vectors via matrix multiplication.

all_context_vecs = attn_weights@inputs
# print(all_context_vecs)
# tensor([[0.4421, 0.5931, 0.5790],
#         [0.4419, 0.6515, 0.5683],
#         [0.4431, 0.6496, 0.5671],
#         [0.4304, 0.6298, 0.5510],
#         [0.4671, 0.5910, 0.5266],
#         [0.4177, 0.6503, 0.5645]])



"""
*Implementing self-attention with trainable weights.

Three trainable weight matrices: Wq,Wk,Wv to project the embedded input tokens x^i into query, key, and value vectors.

"""
x_2 = inputs[1] 
d_in = inputs.shape[1]
d_out = 2    # GPT-like models havs the same input and output dimensions


#Initialize three weight matrices: Wq,Wk,Wv.
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

#requires_grad=False to reduce clutter in the outputs for illustration porposes

#*Step 1: Compute query, key, and value vectors.
query_2 = x_2@W_query
key_2 = x_2@W_key
value_2 = x_2@W_value
# print(query_2) #tensor([0.4306, 1.4551])

#Attention weights determine the extend to which a context vector depends on the different part of the input
#i.e., to what extent the network focuses on different parts of the input.

#Weight parameters are the fundamental, learned coefficients that define the network's connections, while attention weights
# are dynamic, context_specific values.

#Obtain all keys and values via matrix multiplication.
keys = inputs@W_key
values = inputs@W_value
# print("keys.shape: ", keys.shape) #keys.shape:  torch.Size([6, 2])
# print("values.shape: ", values.shape) #values.shape:  torch.Size([6, 2])

#*Step2: Compute the attention score.
#compute dot-product using query and key obtained by transforming the inputs via the respective weight matrices.

keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
# print(attn_score_22) #tensor(1.8524)

#Generalize the computation to all attention scores via matrix multiplication
attn_scores_2 = query_2@keys.T #All attention scores for the given query
# print(attn_scores_2) #tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])

#*Step 3: Going from attention scores to the attention weights.
#Normalize the attention scores w using softmax function to obtain attention wights alpha
#Scale the attention scores by dividing them by the square root(mathematically exponentiating by 0.5) of the embedding dimension of the keys

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2/d_k**0.5, dim=-1)
# print(attn_weights_2) #tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])

#The normalization scaling by d^0.5 
	# •	Reduces variance in attention scores → Prevents extreme softmax outputs.
	# •	Smooths out gradients → Avoids exploding/vanishing gradient problems.
	# •	Improves training stability → Helps optimization converge faster.

#Final step:
#Compute context vectors: Multiplying each value vector with its respective attention wight and then summing them to obtain the context vector.

context_vec_2 = attn_weights_2@values
# print(context_vec_2) #tensor([0.3061, 0.8210])

'''
The terms "key", "query", and "value" in the context of attention mechanisms are borrowed from the domain of information
retrieval and databases, where similar concepts are used to store, search, and retrieve information.
'''


"""
*Implementing a compact self-attention Python Class
"""

import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        #Initialize trainable weight matrices (W_query, W_key, W_value).
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x@self.W_key
        queries = x@self.W_query
        values = x@self.W_value
        attn_scores = queries@keys.T
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights@values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
# print(sa_v1(inputs))

'''
tensor([[0.2996, 0.8053],
        [0.3061, 0.8210],
        [0.3058, 0.8203],
        [0.2948, 0.7939],
        [0.2927, 0.7891],
        [0.2990, 0.8040]], grad_fn=<MmBackward0>)
'''

#*Improve SelfAttention_v1 by utilizing nn.Linear layers, which effectively perform matrix multiplication when the bias units are disabled.
#nn.Linear has an optimized weight initialization scheme, contributing to more stable and effective model training.


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries@keys.T
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=1)
        context_vec = attn_weights@values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
# print(sa_v2(inputs))
'''
tensor([[-0.0739,  0.0713],
        [-0.0748,  0.0703],
        [-0.0749,  0.0702],
        [-0.0760,  0.0685],
        [-0.0763,  0.0679],
        [-0.0754,  0.0693]], grad_fn=<MmBackward0>)
'''

#The output from V2 is different from V1 only because they use different initial weights for the weight matrices
#nn.linear uses a more sophisticated weight initialization scheme.


"""
*Hiding futuren works with causal attention/masked attention
A specialized form of self-attention. 
It restricts a model to consider only previous and current inputs in a sequence when processing any given token.
When computing attention scores, the causal attention mechanism ensures that the model only factors in tokens that occur
at or before the current token in the sequence.
"""

#*Applying a causal attention mask
#*Step 1 apply softmax: attention scores (unnormalized)---> attention weights (normalized)
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries@keys.T
attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=1)
# print(attn_weights)

# tensor([[0.1921, 0.1646, 0.1652, 0.1550, 0.1721, 0.1510],
#         [0.2041, 0.1659, 0.1662, 0.1496, 0.1665, 0.1477],
#         [0.2036, 0.1659, 0.1662, 0.1498, 0.1664, 0.1480],
#         [0.1869, 0.1667, 0.1668, 0.1571, 0.1661, 0.1564],
#         [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.1585],
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
#        grad_fn=<SoftmaxBackward0>)

#*Step 2 Mask with 0s above diagonal: attention weights(normalized)--> Masked attention scores (unnormalized)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
# print(mask_simple)
# tensor([[1., 0., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0., 0.],
#         [1., 1., 1., 0., 0., 0.],
#         [1., 1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1., 1.]])

masked_simple = attn_weights*mask_simple
# print(masked_simple)
# tensor([[0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],
#         [0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],
#         [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
#        grad_fn=<MulBackward0>)

#*Step 3 Normalize rows: Masked attention scores (unnormalized)--> Masked attention weights (normalized)

row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple/row_sums
# print(masked_simple_norm)
# tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
#         [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
#         [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
#        grad_fn=<DivBackward0>)

#After masking and renormalization, the distribution of attention weights is as if it was calculated only among 
# the unmasked positions to begin with. This ensures no information leakage from the future tokens. 

#A more efficient way to obtain the masked attention weight matrix in causal attention is to mask the attention scores 
# with negative infinity values before applying the softmax.

#Softmax convert the inputs to probability, when negative infinity velue is in a row, the softmax function treats them
#as zero probability. (mathmatically, because e^-∞ approaches 0)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
# print(masked)
# tensor([[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
#         [0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
#         [0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
#         [0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
#         [0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
#         [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
#        grad_fn=<MaskedFillBackward0>)

attn_weights = torch.softmax(masked/keys.shape[-1]**0.5, dim=1)
# print(attn_weights)
# tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
#         [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
#         [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
#        grad_fn=<SoftmaxBackward0>)

"""
*Masking additional attention weights with dropout.
Dropout is a regularization technique for reducing overfitting by randomly select hidden layer units to ignore during training. 
Dropout is ONLY applied during training and is disabled afterwards.

In Transformer architecture, dropout in the attention mechanism is typically applied in two specific areas:
After calculating the attention scores or after applying the attention weights to the value vectors.  
"""

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(context_length, context_length)
# print(dropout(example))
# tensor([[2., 2., 2., 2., 2., 2.],
#         [0., 2., 0., 0., 0., 0.],
#         [0., 0., 2., 0., 2., 0.],
#         [2., 2., 0., 0., 0., 2.],
#         [2., 0., 0., 0., 0., 2.],
#         [0., 2., 0., 0., 0., 0.]])

#To compensate for the reduction in active elements, the values of the remaining elements are scaled up by the factor of 1/0.5 = 2.

torch.manual_seed(123)
# print(dropout(attn_weights))
# tensor([[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.8966, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.6206, 0.0000, 0.0000, 0.0000],
#         [0.5517, 0.4921, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.4350, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.3327, 0.0000, 0.0000, 0.0000, 0.0000]],
#        grad_fn=<MulBackward0>)

"""
* Implementing a Compact Causal Attention Class
"""

batch = torch.stack((inputs, inputs), dim=0)
# print(batch.shape)
# torch.Size([2, 6, 3])

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        #When use the CausalAttention class in LLM, buffers are automatically moved to the appropriate device 
        # (CPU or GPU) along with our model. This means we don't need to manually ensure these tensors are on the same device 
        # as your model parameters, avoiding device mismatch errors.


    def forward(self, x):
        b, num_tokens, d_in = x.shape #New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries@keys.transpose(1,2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights@values
        return context_vec


torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
# print("context_vecs.shape: ", context_vecs.shape)
# context_vecs.shape:  torch.Size([2, 6, 2])

"""
* Extending Single-head Attention to Multi-head Attention.

Dividing the attention mechanism into multiple "heads", each opearatiing independently.
"""

#* Stacking multiple singl-head attention layers (Intuitively build multi-head attention module)
#Run the attention mechanism multiple times (in parallel) with different, learned linear projections--the results 
#of multiplying the input data (query, key, and value vectors in attention mechanisms) by a weight matrix. 

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1) # combining multiple single-head attention modules sequentially.
        #Can be improved with matrix multiplication.

        #The final output embedding dimension is (d_out*num_heads)
        #Obtain a tensor with num_heads of context vector matrices. 
        #In each context vectoe matrix, the rows represent the context vectors correponding to the tokens, 
        # and the columns correpond to the embedding dimension specified via d_out. 
        # We contactate these context vector matrices along the column dimension. 

torch.manual_seed(123)
context_length = batch.shape[1] #Number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

# print(context_vecs) 
# print("context_vecs.shape: ", context_vecs.shape) 
# tensor([[[-0.4519,  0.2216,  0.4772,  0.1063],
#          [-0.5874,  0.0058,  0.5891,  0.3257],
#          [-0.6300, -0.0632,  0.6202,  0.3860],
#          [-0.5675, -0.0843,  0.5478,  0.3589],
#          [-0.5526, -0.0981,  0.5321,  0.3428],
#          [-0.5299, -0.1081,  0.5077,  0.3493]],

#         [[-0.4519,  0.2216,  0.4772,  0.1063],
#          [-0.5874,  0.0058,  0.5891,  0.3257],
#          [-0.6300, -0.0632,  0.6202,  0.3860],
#          [-0.5675, -0.0843,  0.5478,  0.3589],
#          [-0.5526, -0.0981,  0.5321,  0.3428],
#          [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>)

#context_vecs.shape:  torch.Size([2, 6, 4])
'''
The first dimension of the resulting vector is 2 because inpute texts are duplicated, 
which is why the context vectors are exactly the same for those.
The second dimention refers to the 6 tokens in each input.
The third dimension refers to the 4-dimensional embedding of each token.
'''


#*Implementing multi-head attention with weight splits.

#Combine MultiHeadAttentionWrapper with CausalAttention and implement more efficiently
#Splits the input into multiple heads by reshaping the projected query, key, and value tensors 
# and then combines the results from these heads after computing attention.

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out%num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out//num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        #Only one matrix multiplication used to computer the keys--> MORE EFFICIENT.
        queries, keys, values = self.W_query(x), self.W_key(x), self.W_value(x)

        queries, keys, values = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2), \
                            keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2), \
                            values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        #*Key operation: 
        # split the d_out dimension into num_heads and head_dim, where head_dim = d_out/num_heads
        # This spliting is achieved by using .view method: a tensor of dimensions(b, num_tokens, d_out)
        #is reshaped to dimension (b, num_tokens, num_heads, head_dim)

        #The tensor is then transposed to bring the num_heads dimension before the num_tokens dimension,
        # resulting in a shape of (b, num_heads, num_tokens, head_dim)
        #This transpose is necessary for correctly aligning the queries, keys and values across different heads
        #and performing batched matrix multiplication effectively.

        attn_scores = queries@keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights@values).transpose(1,2)
        #Reshape/flatterned the context vectors into (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # output projection layer after combining the heads. NOT strictly necessary, but is commonly used in LLM architectures. Added here for completeness.
        context_vec = self.out_proj(context_vec)
        return context_vec
    #Starts with a multi-head layer and then internally splits this layer into individual attention heads

a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                    [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])
# print (a @ a.transpose(2,3))

# tensor([[[[1.3208, 1.1631, 1.2879],
#           [1.1631, 2.2150, 1.8424],
#           [1.2879, 1.8424, 2.0402]],

#          [[0.4391, 0.7003, 0.5903],
#           [0.7003, 1.3737, 1.0620],
#           [0.5903, 1.0620, 0.9912]]]])

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, 2)
context_vecs = mha(batch)
# print(context_vecs)
# print("context_vecs.shape: ", context_vecs.shape)

# tensor([[[0.3190, 0.4858],
#          [0.2943, 0.3897],
#          [0.2856, 0.3593],
#          [0.2693, 0.3873],
#          [0.2639, 0.3928],
#          [0.2575, 0.4028]],

#         [[0.3190, 0.4858],
#          [0.2943, 0.3897],
#          [0.2856, 0.3593],
#          [0.2693, 0.3873],
#          [0.2639, 0.3928],
#          [0.2575, 0.4028]]], grad_fn=<ViewBackward0>)
# context_vecs.shape:  torch.Size([2, 6, 2])