import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_all import bag_of_words, tokenize, stemmer_func
from model import NeuralNet
# https://github.com/python-engineer/pytorch-chatbot
#Load the json file
with open('new_intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# Loop through each patterns in our intents to store for training
for intent in intents['intents']:
    tag = intent['tag']
    #Add each to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # Tokenize each word and store it
        w = tokenize(pattern)
        # Add to our words list
        all_words.extend(w)
        # Create intent and pattern pair
        xy.append((w, tag))

# Stem each word and convert it to lower case
ignore_word = ['?', '.', '!']
all_words = [stemmer_func(w) for w in all_words if w not in ignore_word]
# We will now remove the duplicate words and sort then into sets
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Now lets create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X contains bag of words for each of the pattern question
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss takes class labels(like 0,1,2,3,4..) as input, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters set for the training
num_epochs = 400
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

#Now lets create a Pytorch dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
#GPU support to train faster
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Instiante the Neural Network mode as defined in model.py file
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss() #Since we have multi-label we use CrossEntropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #Adam Optimiser to achive global minima faster

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass of NN and calculate Loss at end of each Pass
        outputs = model(words)
        loss = loss_fn(outputs, labels)
        
        # Backward pass of NN and optimize the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

#Create the checkpoint of model weights
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

#File Name to save model, and actual save function
FILE = "data_new_intent.pth"
torch.save(data, FILE)

#Print the succesfull saving of the file after training
print(f'Training is completed. Model Checkpoint file saved to {FILE}')