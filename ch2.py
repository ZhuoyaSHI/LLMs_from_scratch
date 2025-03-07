import re

#Read the text file
with open("the_verdict.txt", 'r', encoding = 'utf-8') as f:
    raw_text = f.read()
# print("Total number of character: ", len(raw_text))
# print(raw_text[:99])

#Tokenization
preprocessed = re.split(r'([,.?_!"{}\(\)\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(len(preprocessed))
# print(preprocessed[:30])

#Determin the vocabulary size
all_words = sorted(list(set(preprocessed)))
vocab_size = len(all_words)
# print(vocab_size)

#Creating a vocabulary 
vocab = {token: integer for integer, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i > 50:
#         break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s , i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"{}\(\)\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]  
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}

# print(len(vocab.items()))
# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s , i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"{}\(\)\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]  
        preprocessed = [item if item in self.str_to_int else '<|unk|>' for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = "<|endoftext|>".join((text1, text2))
# print(text)

tokenizer = SimpleTokenizerV2(vocab)
# print(tokenizer.encode(text))
# print(tokenizer.decode(tokenizer.encode(text)))

# from importlib.metadata import version 
import tiktoken
# print("tiktoken version: ", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode(text, allowed_special = {"<|endoftext|>"})
# print(integers)

strings = tokenizer.decode(integers)
# print(strings)

with open("the_verdict.txt", "r", encoding = 'utf-8') as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))
enc_sample = enc_text[50:] #

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
# print(f"x: {x}")
# print(f"y:      {y}")


for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    # print(tokenizer.decode(context), "------>", tokenizer.decode([desired]))

'''
Dataset for bactched inputs and targets
'''

import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunck = token_ids[i : i + max_length]
            target_chunck = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunck))
            self.target_ids.append(torch.tensor(target_chunck))
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(text, batch_size = 4, max_length = 256, stride = 128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

with open("the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# # print(first_batch)
# second_batch = next(data_iter)
# # print(second_batch)

# dataloader = create_dataloader_v1(raw_text, batch_size = 8, max_length=4, stride=4,shuffle=False)

# data_iter = iter(dataloader)
# inputs, targets = next(data_iter)
# print("Input: \n", inputs)
# print("\nTarget: \n", targets)
'''
max_length is the same as the stride. 
Utilize the data set fully but avoid any overlap between the batches, since more overlap could lead to increased overfitting.
'''

output_dim = 256
vocab_size = 50257
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

'''
Embedding tokenIDs into 256 dimentional vectors.
'''
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter =  iter(dataloader)
inputs,targets = next(data_iter)
# print("Token IDs:\n", inputs)
# print("\nInputer shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
# print(token_embeddings.shape)
'''
Create another embedding layer that has the same dimension as the token_embedding_layer.
'''
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
# print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
# print(input_embeddings.shape)