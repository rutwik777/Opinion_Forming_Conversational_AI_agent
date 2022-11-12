import random
import json
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline 
from model import NeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "sentence-transformers/all-MiniLM-L6-v2"
bert_model = SentenceTransformer(model_name)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer_summarization = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model_summarization = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
pipe_summary = pipeline("summarization", model=model_summarization, tokenizer=tokenizer_summarization)

with open('new_intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "bert_data_new_intent.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

tags = []
bot_name = "Chitti"
print("Let's chat! (type 'quit' to exit)")
print("I can converse on folowing topics ")
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
print(tags)
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    X = bert_model.encode(sentence)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(prob)
    if prob.item() > 0.40:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                bot_output = random.choice(intent['responses'])
                if (len(bot_output.split())) >=40:
                    output_summary = pipe_summary(bot_output)
                    output_summary = output_summary[0]["summary_text"]
                    print(f"{bot_name}: {output_summary}")
                else:
                    print(f"{bot_name}: {bot_output}")
    else:
        print(f"{bot_name}: I do not understand...")