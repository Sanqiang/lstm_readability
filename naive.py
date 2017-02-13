# wordvector-based
import os
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from functools import reduce
import pickle


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
    return sum([(vec1[i] - vec2[i])**2 for i in range(len(vec1))])

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
            data_x[category_clean].append((word_per_sent))
            data_sx.append((word_per_sent))

            seg_per_sent = 0
            for sent in text:
                for i in range(1, len(sent)):
                    word = sent[i]
                    word_pre = sent[i - 1]
                    if word not in glove_vector or word_pre not in glove_vector or compareVector(glove_vector[word], glove_vector[word_pre]) < .5:
                        seg_per_sent += 1
            seg_per_sent /= len(text)
            data_x2[category_clean].append((seg_per_sent))
            data_sx2.append((seg_per_sent))

            data_y[category_clean].append(int(label))
            data_sy.append(int(label))


f_data_x2 = open("data_x2", data_x2)
pickle.dump(data_x2, f_data_x2)

f_data_sx2 = open("data_sx2", data_sx2)
pickle.dump(data_sx2, f_data_sx2)

f_data_x = open("data_x", data_x)
pickle.dump(data_x, f_data_x)

f_data_sx = open("data_sx", data_sx)
pickle.dump(data_sx, f_data_sx)

f_data_y = open("data_y", data_y)
pickle.dump(data_y, f_data_y)

f_data_sy = open("data_sy", data_sy)
pickle.dump(data_sy, f_data_sy)

reg = linear_model.Ridge(alpha = 1.0)
scores = cross_val_score(reg, data_sx, data_sy, cv=10, n_jobs=-1, verbose=0)
print(reduce(lambda x, y: x + y, scores) / len(scores))

scores = cross_val_score(reg, data_sx2, data_sy, cv=10, n_jobs=-1, verbose=0)
print(reduce(lambda x, y: x + y, scores) / len(scores))


