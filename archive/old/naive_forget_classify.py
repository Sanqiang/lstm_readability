import os
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from functools import reduce
import pickle
import numpy as np
import math

home = os.environ["HOME"]
thresholds = [threshold/10.0 for threshold in range(10)]


data = []
for line in open("forget_label.txt"):
    if len(line.strip()) > 0:
        data.append((line.split("\t")))

num_label = 5
all_labels = []
for entry in data:
    label = int(entry[1])
    all_labels.append(label)
sorted(all_labels)

thres_label = []
step = 1.0 / num_label
while step < 1:
    thres_label.append(all_labels[step])
    step *= 2
thres_label.remove(thres_label[-1])

for threshold in thresholds:
    datas_category = {}
    datas = []

    labels_category = {}
    labels = []

    for entry in data:
        sims = [[float(sim) for sim in sent.split()] for sent in entry[0].split("\v")]
        label = int(entry[1])
        # if label > 5000:
        #     continue
        category = entry[2]

        if category not in datas_category:
            datas_category[category] = []
            labels_category[category] = []

        num_seg = 1
        for sent in sims:
            #num_seg += 1
            for sim in sent:
                if sim < threshold:
                    num_seg += 1
        num_seg /= len(sims)

        datas.append((num_seg, num_seg))
        datas_category[category].append((num_seg, num_seg))

        i = 0
        while i < len(thres_label) and label > thres_label[i]:
            i += 1

        labels.append(i)
        labels_category[category].append(i)

    # reg = linear_model.Ridge(alpha=0.0)
    reg = LogisticRegression()
    scores = cross_val_score(reg, datas, labels, cv=10, n_jobs=-1, verbose=0)

    score = reduce(lambda x, y: x + y, scores) / len(scores)
    #print(score)

    for origin_category in datas_category:
        for target_category in datas_category:
            reg.fit(datas_category[origin_category], labels_category[origin_category])

            score = reg.score(datas_category[target_category], labels_category[target_category])

            print("\t".join([origin_category.strip(), target_category.strip(), str(score)]))
