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

''' The token embedding is handled inside the GPT model. 
 In LLM, the embedded input token dimension typically matches the output dimension'''

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

#*Normalizeing activations with layer normalization

'''The main idea of layer normalization is to adjust the activations (outputs) of a neural network layer to have a mean of 0 and a variance of 1, also known as unit variance.
    --> speed up the convergence to effective weights and ensures consistent, reliable training.

     Layer Normalization is typically applied before and after the multi-head attention module and before the final output layer. '''

torch.manual_seed(123)
batch_example = torch.randn(2,5)
layer = nn.Sequential(nn.Linear(5,6), nn.ReLU()) #RELU: thresholds negative inputs to 0, ensuring that a layer outputs only positive values.abs
out = layer(batch_example)
# print(out)
# tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
#         [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
#        grad_fn=<ReluBackward0>)



mean = out.mean(dim=-1, keepdim=True) #keepdim=True ensures the output tensor retains the same shape as the input tensor, even though the operation reduces the tensor along the dimension specified via dim. 
var = out.var(dim=-1, keepdim=True) #dim specifies the dimension along which the calculation is performed in a tensor. for 2D tensor, dim=1 or dim=01--> across the column dimension to obtain one mean per row; dim=0 --> across the row dimension to obtain one mean per column. 
# print("Mean:\n", mean)
# print("Variance:\n", var)
# Mean:
#  tensor([[0.1324],
#         [0.2170]], grad_fn=<MeanBackward1>)
# Variance:
#  tensor([[0.0231],
#         [0.0398]], grad_fn=<VarBackward0>)

out_norm = (out - mean)/torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True) 
var = out_norm.var(dim=-1, keepdim=True)
# print("Normalized layer outputs:\n", out_norm)
# print("Mean:\n", mean)
# print("Variance:\n", var)
#Mean:
#  tensor([[-5.9605e-08],
#         [ 1.9868e-08]], grad_fn=<MeanBackward1>)
# Variance:
#  tensor([[1.0000],
#         [1.0000]], grad_fn=<VarBackward0>)

torch.set_printoptions(sci_mode=False)
# print("Mean:\n", mean)
# print("Variance:\n", var)
# Mean:
#  tensor([[    -0.0000],
#         [     0.0000]], grad_fn=<MeanBackward1>)
# Variance:
#  tensor([[1.0000],
#         [1.0000]], grad_fn=<VarBackward0>)


#*Normalization operates on the last dimension of the input tensor. 
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  #small constant (epsilon) added to the variance to prevent division by 0 during the normalization. 
        self.scale = nn.Parameter(torch.ones(emb_dim)) #Trainable patameters
        self.shift = nn.Parameter(torch.zeros(emb_dim)) #Trainable patameters

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) #DOES NOT apply with Bessel's correction. Because in LLM, the difference between using n and n-1 in the denominator is practically negligible. 
        norm_x = (x-mean)/torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
# print("Mean:\n", mean)
# print("Variance:\n", var)
# Mean:
#  tensor([[    -0.0000],
#         [     0.0000]], grad_fn=<MeanBackward1>)
# Variance:
# Variance:
#  tensor([[1.0000],
#         [1.0000]], grad_fn=<VarBackward0>)

'''
NOTE: Compare Layer Normalization versus Batch Normalization.

Batch normalization: normalizes across the batch dimension.
Layer normalization: normalizes across the feature dimension. 
LLM requires significant computational resources, and available hardware or specific use case can dictate the batch size during training and inference.
Layer normalization normalizes each input independently of the batch size--> offers more flexibility and stability in these scenario.
--> Benificial for distributed training or deploying models in resources constrained environments.
'''

#* Implementing a feed forward network with GELU (Gaussian Error Linear Unit) activation.

'''GELU and SwiGLU(sigmoid-werighted linear unit) activation are more complex and smoothe than ReLU, incorporating Gaussian and sigmoid-gated linear units.  
Offer improved performance for deep learning models.'''

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

#GELU(x)=x Φ(x), where Φ(x) is the cumulative distribution function of the standard Gaussian distribution.
#Implement a computationally cheaper approximation. (GPT2)
# GELU(x) ≈ 0.5 ⋅ x ⋅ (1 + tanh[√((2/π)) ⋅ (x + 0.044715 ⋅ x^3])

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

import matplotlib.pyplot as plt
gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU, ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
# plt.show()

#The smoothness of GELU leads to better optimization properties during training. 
# --> Allows more nuanced adjustments to the model parameters. 
#RELU outputs 0 for all negative inputs, while GELU allows a small, non-zero output for negative values. 
# --> During training, neurons receive negative input can still contribute to the learning process, though lesser compared to the positive inputs. 

class FeedForward(nn.Module):

    #Small neural network consisting of two linear layers and one GELU activation function. 
    #It receives input batches with tokens that have an embedding size of 768 via GPT_CONFIG_124M dictionary where GPT_CONFIG_124M["emb_dim"] = 768. 
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            (nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"])), #increase embedding dimension by  a factor of 4. e.g.,  (2, 3, 768)--> (2, 3, 3072)
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]), #decrease embedding dimension by a factor of 4. e.g.,  (2, 3, 3072)--> (2, 3, 768)
        )

    def forward(self, x):
        return self.layers(x)

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2,3,768)
out=ffn(x)
# print(out.shape) # torch.Size([2, 3, 768])

"""
NOTE: 
This FeedForward Module enhancing the model's ability to learn from and generalize the data.
Although the input and output dimensions of this module are the same, it internally expands the embedding dimension into a higher-dimensional space 
through the first linear layer. This expansion is followed by a non-linear GELU activation, and then a contraction back to the original dimension 
with the second linear transformation. Such a design allows for the exploration of a richer representation space.

PLUS, the uniformity in input and output dimensions simplifies the architecture by enabling the stacking of multiple layers, 
without the need to adjust dimensions between them, thus making the model more scalable.
"""

#*Adding shortcut connections

'''Shortcut connections are used to mitigate the challenge of vanishing gradients

Vanishing gradient: gradients (which guide weight updates during training) become progressively smaller as they propogate backward through the layers.
Making it difficult to effectively train earlier layers.

Shortcut connections: adding inputs of a layer to its outputs, effectively creating an alternate path that bypasses certain layers. 
Helps maintain relatively large gradiant values even in early layers.
'''

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            #Implement 5 layers
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            #Compute the output of current layer
            layer_output=layer(x)
            #If using shortcut, add the input to the output
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

layer_sizes = [3,3,3,3,3,1]
sample_inputs = torch.tensor([[1.,0.,-1.]]) #Keep the float format
torch.manual_seed(123) #specify random seed for initial weights for reproducibility. 
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

def print_gradients(model, x):
    #forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    #calculate loss based on how close the target and output are
    loss = nn.MSELoss()
    loss = loss(output, target)

    #Backward pass to calculate the gradients 
    loss.backward() #convenient method in PyTorch that computes loss gradients, which are required during training

    #print the gradients of each parameter in the model
    for name, param in model.named_parameters():
        if 'weight' in name:
            #Print the mean absolute gradient of the weights
            print(f" {name} has gradient mean of {param.grad.abs().mean().item()}")

# print_gradients(model_without_shortcut, sample_inputs)
# torch.Size([2, 3, 768])
#  layers.0.0.weight has gradient mean of 0.00020173587836325169
#  layers.1.0.weight has gradient mean of 0.0001201116101583466
#  layers.2.0.weight has gradient mean of 0.0007152041071094573
#  layers.3.0.weight has gradient mean of 0.0013988735154271126
#  layers.4.0.weight has gradient mean of 0.005049645435065031

'layers.4 to layers.0, the gradient becomes smaller --> Vanishing gradient problem'

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
# print_gradients(model_with_shortcut, sample_inputs)

# torch.Size([2, 3, 768])
#  layers.0.0.weight has gradient mean of 0.22169791162014008
#  layers.1.0.weight has gradient mean of 0.20694106817245483
#  layers.2.0.weight has gradient mean of 0.32896995544433594
#  layers.3.0.weight has gradient mean of 0.2665732204914093
#  layers.4.0.weight has gradient mean of 1.3258540630340576

'layers.4 still has larger gradient mean, BUT the gradient values stablizes as moving towards the layers.0. Does NOT shrink to a vanishingly small number.'

#* Connecting attention and linear layers in a transformer block

'''The operation within the transformer block are designed to keep transform there vectors in a way to preserve their dimentions.
Self-attention mechanism in the multi-head attention block identifies & analyzes relationships between elements in the input sequence.
Feed forward network modified the data individually at each position.'''

from ch3 import  MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A Transformer block that includes multihead attention mechanism and feed forward network, both configured based on provided configuration dictionary(cfg).

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            # block_size=cfg['context_length'],
            context_length=cfg['context_length'],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.ff=FeedForward(cfg)

        #Pre-LayerNorm. 
        #Layer normalization is applied before each of these two components
        self.norm1=LayerNorm(cfg["emb_dim"])
        self.norm2=LayerNorm(cfg["emb_dim"])
        #Dropout is applied after them regularize the model and prevent overfitting. 
        self.drop_resid=nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        #Each component is followed by a shortcut connection that adds the input of the block to its output.
        #--> Helps gradient flow through the network during training, improve the learning of deep models. 
        shortcut=x
        x=self.norm1(x)
        x=self.att(x)
        x=self.drop_resid(x)
        x=x+shortcut

        shortcut=x
        x=self.norm2(x)
        x=self.ff(x)
        x=self.drop_resid(x)
        x=x+shortcut

        return x
    
torch.manual_seed(123)
x=torch.rand(2,4,768)
block=TransformerBlock(GPT_CONFIG_124M)
output=block(x)

# print("Input shape: ", x.shape)
# print("Output shape: ", output.shape) 
# Input shape:  torch.Size([2, 4, 768])
# Output shape:  torch.Size([2, 4, 768])

"""
NOTE: The transformer architecture processes sequence without altering their shape--> Crucial designe. 
Enables effective application across a wide range of sequence-to-sequence  tasks, where each output vector directly corresponds to an input vector, mantening an one-to-one relationship. 
Output is a context vector that encapsulates information from the entire input vector. 
"""

#* Coding the GPT model

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #initialize token and position embedding layers using configurations passed in via cfg 
        self.tok_emb=nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb=nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb=nn.Dropout(cfg["drop_rate"])

        self.trf_block=nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm=LayerNorm(cfg["emb_dim"]) #Standardize the outputs from transformer blocks, stablize the learning process
        #linear output without bias, projects the transformer's output into the vocabulary space of the tokennizer to genetate logits for each token in the vpcabulary
        self.out_head=nn.Linear(cfg["emb_dim"],cfg["vocab_size"], bias=False) 

    def forward(self, in_idx):
        batch_size, seq_len=in_idx.shape
        tok_embeds=self.tok_emb(in_idx)
        pos_embeds=self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x=tok_embeds + pos_embeds

        x=self.drop_emb(x)
        x=self.trf_block(x)
        x=self.final_norm(x)
        logits=self.out_head(x) #representing the next token's unnormalized probability. Will need to be converted into tokens and text outputs later. 
        return logits

torch.manual_seed(123)
model=GPTModel(GPT_CONFIG_124M)

out=model(batch)
# print("Input batch:\n", batch)
# print("\nOutput shape:\n", out.shape)
# print(out)

# Input batch:
#  tensor([[6109, 3626, 6100,  345],
#         [6109, 1110, 6622,  257]])

# Output shape:
#  torch.Size([2, 4, 50257])
# tensor([[[ 0.3613,  0.4223, -0.0711,  ...,  0.3483,  0.4661, -0.2838],
#          [-0.1792, -0.5660, -0.9485,  ...,  0.0477,  0.5181, -0.3168],
#          [ 0.7120,  0.0332,  0.1085,  ...,  0.1018, -0.4327, -0.2553],
#          [-1.0076,  0.3418, -0.1190,  ...,  0.7195,  0.4023,  0.0532]],

#         [[-0.2564,  0.0900,  0.0335,  ...,  0.2659,  0.4454, -0.6806],
#          [ 0.1230,  0.3653, -0.2074,  ...,  0.7705,  0.2710,  0.2246],
#          [ 1.0558,  1.0318, -0.2800,  ...,  0.6936,  0.3205, -0.3178],
#          [-0.1565,  0.3926,  0.3288,  ...,  1.2630, -0.1858,  0.0388]]],
#        grad_fn=<UnsafeViewBackward0>)

#Collect the total numver of parameters in the model's parameter tensors. 
total_params=sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}")
# Total number of parameters: 163,009,536 --> NOT 124 million of GPT2 

"""
*Weight Tying: 
The original GPT2 architecture is reusing the weights from the token embedding layer inits output layer.
--? Reducing the overall memory footprint and computational complexity of the model.
NOTE: Using separate token embedding and output layers results in better training and model performance. 
"""

# print("Token embedding layer shape:", model.tok_emb.weight.shape)
# print("Output layer shape:", model.out_head.weight.shape)
# Token embedding layer shape: torch.Size([50257, 768])
# Output layer shape: torch.Size([50257, 768])

total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
# print(f"Number of trainable parameters considering the weight tying: {total_params_gpt2:,}")
# Number of trainable parameters considering the weight tying: 124,412,160 --> BINGO!

#Calculate and compare the number of parameters that are contained in the feed forward module and those are contained in the multi-head attention module
total_size_bytes=total_params*4 #Each parameter is a 32-bit float taking up 4 bytes 
total_size_mb = total_size_bytes/(1024*1024)
# print(f"Total size of the model: {total_size_mb:.2f} MB")
# Total size of the model: 621.83 MB

#*Generating Text
'''
- Decoding the output tensors
- Selecting tokens based on a probability distribution (softmax function)
- COnverting these tokens into human-readable text.
The model generates the most likely next token [Greedy Decoding].
'''

def generate_text_simple(model, idx, max_new_tokens, context_size):
    #Iterates for a specified number of new tokens to be generated
    for _ in range(max_new_tokens):
        #Crops the current text to fit the model;s maximum text size
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            #computes predictions
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        #convert logits to probabilities
        #Monotonic softmax function--> Preserve the order of its inputs when transformed into outputs. 
        #[Redundant in practice] because the position with the highest score in softmax output is the same in the logit tensor. 
        #--> Apply torch.argmax directly to the logits. 
        probas = torch.softmax(logits, dim=-1)
        #select next token based on the highest probability prediction. 
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

#Step1: Encode the input context into token IDs
start_text = "Hello, I am"
encoded = tokenizer.encode(start_text)
# print("encoded: ", encoded) # encoded:  [15496, 11, 314, 716]
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
# print("encoded_tensor.shape: ", encoded_tensor.shape)  # encoded_tensor.shape:  torch.Size([1, 4])

#Step2: put model into .eval() mode, disables random components like dropouts, which are only used during training.

model.eval()

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
# print("Output: ", out)
# print("Output length: ", len(out[0]))
# Output:  tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267]])
# Output length:  10

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
# print(decoded_text)
# Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous

#Not trained yet, so just generating coherent text. 