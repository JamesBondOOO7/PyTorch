import torch
import torch.nn as nn
import torchtext.legacy.data as ttd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Let's make the fake data
data = {
    "label": [0, 1, 1],
    "data": [
        "I like eggs and ham.",
        "Eggs I like!",
        "Ham and eggs or just ham?"
    ]
}

df = pd.DataFrame(data)

print(df.head())
df.to_csv('thedata.csv', index=False)

# Creating Field Objects

TEXT = ttd.Field(
    sequential=True,
    batch_first=True,
    lower=True,
    tokenize='spacy',  # by default : string.split()
    pad_first=True
)

LABEL = ttd.Field(
    sequential=False,
    use_vocab=False,
    is_target=True
)

# Note: if you don't specify use_vocab=False, then PyTorch will
# complain later when you try to iterate over the dataset that
# the attribute `vocab` doesn't exist.

# Note 2: if you don't specify is_target=True, then PyTorch will
# assume it's part of the input, so when you iterate over the
# dataset it will be like:
# for (inputs, targets), _ in iterator:
# where the 2nd element (_) should have been the target.

dataset = ttd.TabularDataset(
    path="thedata.csv",
    format='csv',
    skip_header=True,
    fields=[('label', LABEL), ('data', TEXT)]
)

print(dataset.examples[0])
print(type(dataset.examples[0]))
print(dataset.examples[0].data)
print(dataset.examples[0].label)

# Train test split
train_dataset, test_dataset = dataset.split(0.66)  # by default = 0.7

for ex in train_dataset.examples:
    print(ex.data)

# Build the Vocab
TEXT.build_vocab(train_dataset,)

vocab = TEXT.vocab
print(type(vocab))

print(vocab.stoi)  # word-index mapping
print(vocab.itos)  # list of reverse mapping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Iterator for loading data
train_iter, test_iter = ttd.Iterator.splits(
    (train_dataset, test_dataset), sort_key=lambda x: len(x.data),
    batch_sizes=(2, 2), device=device
)

for inputs, targets in train_iter:

    print("inputs:", inputs, "shape:", inputs.shape)
    print("targets:", targets, "shape:", targets.shape)
    break

for inputs, targets in test_iter:
    print("inputs:", inputs, "shape:", inputs.shape)
    print("targets:", targets, "shape:", targets.shape)
    break
