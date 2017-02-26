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

        if label >= 700:
            labels.append(1)
            labels_category[category].append(1)
        else:
            labels.append(0)
            labels_category[category].append(0)

        # labels.append(label)
        # labels_category[category].append(label)

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
