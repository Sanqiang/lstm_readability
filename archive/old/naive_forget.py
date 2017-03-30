# record forgate in naive approach

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

print(len(glove_vector))

#utility

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

forget_gate_f = open("forget_label.txt", "w")
forget_data = ""

#data
path_data = "".join([home, "/data/newsela/news3.txt"])
handler = open(path_data, "r")
obj = json.load(handler)
data_x = {}
data_x2 = {}
data_y = {}
data_sx = []
data_sy = []
data_sx2 = []
for category in obj:
    category_clean = category.strip()
    if category_clean not in data_x:
        print(category_clean)
        data_x[category_clean] = []
        data_x2[category_clean] = []
        data_y[category_clean] = []

    data_entries = obj[category]
    for data_entry in data_entries:
        for label in data_entry:

            forget_line = ""
            text = data_entry[label]

            text = [word_tokenize(sent) for sent in sent_tokenize(text)]
            #avg sen
            word_per_sent = 0
            for sent in text:
                word_per_sent += len(sent)
            word_per_sent /= len(text)
            data_x[category_clean].append((word_per_sent, word_per_sent))
            data_sx.append((word_per_sent, word_per_sent))

            seg_per_sent = 0
            for sent in text:
                for i in range(1, len(sent)):
                    word = sent[i]
                    word_pre = sent[i - 1]

                    gate = 0
                    if word in glove_vector and word_pre in glove_vector:
                        gate = compareVector(glove_vector[word], glove_vector[word_pre])
                        forget_line = " ".join([forget_line, str(gate)])

                    forget_line = "".join([forget_line, "\v"])
                    if word not in glove_vector or word_pre not in glove_vector or compareVector(glove_vector[word], glove_vector[word_pre]) < .5:
                        seg_per_sent += 1
            seg_per_sent /= len(text)
            data_x2[category_clean].append((seg_per_sent, seg_per_sent))
            data_sx2.append((seg_per_sent, seg_per_sent))

            data_y[category_clean].append(int(label))
            data_sy.append(int(label))

            forget_line = "\t".join([forget_line, label, category])
            forget_data = "\n".join([forget_data , forget_line])

        if len(forget_data) > 100000:
            forget_gate_f.write(forget_data)
            forget_data = ""

forget_gate_f.write(forget_data)
forget_gate_f.close()

