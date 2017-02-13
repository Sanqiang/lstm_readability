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
                    if word not in glove_vector or word_pre not in glove_vector or compareVector(glove_vector[word], glove_vector[word_pre]) < .5:
                        seg_per_sent += 1
            seg_per_sent /= len(text)
            data_x2[category_clean].append((seg_per_sent, seg_per_sent))
            data_sx2.append((seg_per_sent, seg_per_sent))

            data_y[category_clean].append(int(label))
            data_sy.append(int(label))



f_data_x2 = open("data_x2", "wb")
pickle.dump(data_x2, f_data_x2)

f_data_sx2 = open("data_sx2", "wb")
pickle.dump(data_sx2, f_data_sx2)

f_data_x = open("data_x", "wb")
pickle.dump(data_x, f_data_x)

f_data_sx = open("data_sx", "wb")
pickle.dump(data_sx, f_data_sx)

f_data_y = open("data_y", "wb")
pickle.dump(data_y, f_data_y)

f_data_sy = open("data_sy", "wb")
pickle.dump(data_sy, f_data_sy)

result = open("result.5.txt", "a")

reg = linear_model.Ridge(alpha = 1.0)
scores = cross_val_score(reg, np.array(data_sx), np.array(data_sy), cv=10, n_jobs=-1, verbose=0)
score = reduce(lambda x, y: x + y, scores) / len(scores)
result.write("ave sen \t")
result.write(str(score))
result.write("\n")

scores = cross_val_score(reg, data_sx2, data_sy, cv=10, n_jobs=-1, verbose=0)
score = reduce(lambda x, y: x + y, scores) / len(scores)
result.write("ave seg \t")
result.write(str(score))
result.write("\n")

result.write("==================================================\n")

for origin_category in data_x:
    for target_category in data_x:
        result.write("from\t")
        result.write(origin_category)
        result.write("\tto\t")
        result.write(target_category)
        result.write("\n")

        reg = linear_model.Ridge(alpha=1.0)
        reg.fit(data_x[origin_category], data_y[origin_category])
        score = reg.score(data_x[target_category], data_y[target_category])
        result.write("ave sen \t")
        result.write(str(score))
        result.write("\n")

        reg.fit(data_x2[origin_category], data_y[origin_category])
        score = reg.score(data_x2[target_category], data_y[target_category])
        result.write("ave seg \t")
        result.write(str(score))
        result.write("\n")

        result.write("==================================================\n")







