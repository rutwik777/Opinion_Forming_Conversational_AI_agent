import random
import json
import torch
from model import NeuralNet
from nltk_all import bag_of_words, tokenize
#https://github.com/python-engineer/pytorch-chatbot
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Load the json file
with open('new_intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data_new_intent.pth"
data = torch.load(FILE)

#Loading model state and parameters
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

#Give out bot a name, inspired by AI's name in Indian sci-fi movie ROBOT
bot_name = "Chitti" 
def get_response(msg):
    while True:
        if msg == "quit":
            break
        else: 
            sentence = tokenize(msg)
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)
            output = model(X)
            _, predicted = torch.max(output, dim=1)
            tag = tags[predicted.item()]
            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        return random.choice(intent['responses'])
            
            return "I do not understand..."

print("Let's chat! (Please type 'quit' to exit)")   