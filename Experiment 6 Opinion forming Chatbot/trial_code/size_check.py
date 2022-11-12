import numpy as np
import random
import json
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

model_name = "sentence-transformers/all-MiniLM-L6-v2"
bert_model = SentenceTransformer(model_name)

with open('new_intents.json', 'r') as f:
    intents = json.load(f)

tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        w = pattern
        # add to xy pair
        xy.append((w, tag))

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bert_model.encode(pattern_sentence)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(len(X_train[0]))
print(len(y_train))