import random
import json
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from model import NeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "sentence-transformers/all-MiniLM-L6-v2"
bert_model = SentenceTransformer(model_name)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
tokenizer_summarization = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model_summarization = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
pipe_summary = pipeline("summarization", model=model_summarization, tokenizer=tokenizer_summarization)

with open('new_intents.json', 'r') as json_data:
    intents = json.load(json_data)

questions = []
for intent in intents['intents']:
    for pattern in intent["patterns"]:
        questions.append(pattern)
print(questions)    
que_vectors = bert_model.encode(questions)
print (que_vectors)

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

def retrieveSimilarQuestion(question_embedding,sentence_embeddings,sentences):
    max_sim=-1;
    index_sim=-1;
    for index,faq_embedding in enumerate(sentence_embeddings):
        sim=cosine_similarity(faq_embedding.reshape(1, -1),question_embedding.reshape(1, -1))[0][0];
        #sim=cosine_similarity(faq_embedding,question_embedding)[0][0];
        #print(index, sim, sentences[index])
        if sim>max_sim:
            max_sim=sim;
            index_sim=index;
    return (sentences[index_sim])

bot_name = "Chitti"
print("Let's chat! (type 'quit' to exit)")
print("I can converse on folowing topics ")
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
print(tags)
def get_response(msg):
    while True:
        if msg == "quit":
            break
        else:
            # sentence = "do you use credit cards?"
            sentence = msg
            if sentence == "quit":
                break

            X = bert_model.encode(sentence)
            similar_X = retrieveSimilarQuestion(X, que_vectors, questions)
            print(similar_X)
            similar_X_encoded = bert_model.encode(similar_X)
            X = similar_X_encoded.reshape(1, similar_X_encoded.shape[0])
            X = torch.from_numpy(X).to(device)

            output = model(X)
            _, predicted = torch.max(output, dim=1)

            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            print(prob)
            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        bot_output = random.choice(intent['responses'])
                        if (len(bot_output.split())) >=40:
                            output_summary = pipe_summary(bot_output)
                            output_summary = output_summary[0]["summary_text"]
                            return output_summary
                        else:
                            return bot_output
                        #return random.choice(intent['responses'])
            else:
                return "I do not understand..."