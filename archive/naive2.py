# wordvector-based
import os
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from functools import reduce
import pickle
import numpy as np
import math


home = os.environ["HOME"]
#process glove
glove_vector = {}
glove_path = "".join([home, "/data/glove/glove.twitter.27B.200d.txt"])

for line in open(glove_path):
    item = line.split()
    glove_vector[item[0]] = [float(val) for val in item[1:]]

def compareVector(vec1, vec2):
    #norm
    norm_vec1 = 0.0
    for val in vec1:
        norm_vec1 += val * val
    norm_vec1 = math.sqrt(norm_vec1)

    norm_vec2 = 0.0
    for val in vec2:
        norm_vec2 += val * val
    norm_vec2 = math.sqrt(norm_vec2)

    sim = 0
    for i in range(len(vec1)):
        sim += (vec1[i]/norm_vec1) * (vec2[i]/norm_vec2)

    return sim

while True:
    text = input("your sentence:\n")
    text = [word_tokenize(sent) for sent in sent_tokenize(text)]
    seg_per_sent = 0
    for sent in text:
        for i in range(1, len(sent)):
            word = sent[i].lower()
            word_pre = sent[i - 1].lower()
            print(word, word in glove_vector)
            print(word_pre, word_pre in glove_vector)
            print(compareVector(glove_vector[word], glove_vector[word_pre]))

            if word not in glove_vector or word_pre not in glove_vector or compareVector(glove_vector[word],
                                                                                         glove_vector[word_pre]) < .5:
                seg_per_sent += 1
    seg_per_sent /= float(len(text))

    print(seg_per_sent)
