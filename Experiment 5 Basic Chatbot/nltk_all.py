import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
#https://github.com/python-engineer/pytorch-chatbot
stemmer = PorterStemmer()
def tokenize(sentence):
    #split sentence into array of words
    return nltk.word_tokenize(sentence)


def stemmer_func(word):
    #stemming = get the root form of the word
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    #return bag of words array: 1 for each known word that exists in the sentence, else 0
    sentence_words = [stemmer_func(word) for word in tokenized_sentence]
    # initialize bag corpus
    bag_corpus = np.zeros(len(words), dtype=np.float32)
    for id, w in enumerate(words):
        if w in sentence_words: 
            bag_corpus[id] = 1

    return bag_corpus