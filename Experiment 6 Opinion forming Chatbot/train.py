import numpy as np
import random
import json
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#https://github.com/python-engineer/pytorch-chatbot
from model import NeuralNet

#Load the SBERT based sentence transfomer model to create word embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
bert_model = SentenceTransformer(model_name)

#Load the json file
with open('new_intents.json', 'r') as f:
    intents = json.load(f)

tags = []
xy = []
# Loop through each patterns in our intents to store for training
for intent in intents['intents']:
    tag = intent['tag']
    #Add each to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        w = pattern
        # Create intent and pattern pair
        xy.append((w, tag))

# Now lets create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X contains SBERT word embedding for each of the pattern question
    bag = bert_model.encode(pattern_sentence)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss takes class labels(like 0,1,2,3,4..) as input, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters set for the training
num_epochs = 500
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
criterion = nn.CrossEntropyLoss() #Since we have multi-label we use CrossEntropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #Adam Optimiser to achive global minima faster

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass of NN and calculate Loss at end of each Pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
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
"tags": tags
}

#File Name to save model, and actual save function
FILE = "bert_data_new_intent.pth"
torch.save(data, FILE)

#Print the succesfull saving of the file after training
print(f'training complete. file saved to {FILE}')