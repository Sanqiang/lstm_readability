import os
import numpy as np
from data import DataProvider
from gensim.models.word2vec import Word2Vec


tag = "test_addneg_simple"
home = os.environ["HOME"]

if True:
    model = Word2Vec.load_word2vec_format("temp.txt")
    while True:
        word = input("input\n")
        ll = model.most_similar(word)
        for ww in ll:
            print(ww)

if False:
    log_path = "/".join([home,"data","model", tag])
    model_path = "/".join([log_path, "model.txt.npy"])

    embedding = np.load(model_path)
    embedding = embedding[0]
    path_news_dict = "".join([home,
                              "/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/dict2.txt"])
    f = open(path_news_dict, "r")
    word2idx = {}
    idx2word = []
    for line in f:
        words = line.split()
        for word in words:
            word = word.strip()
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                idx2word.append(word)

    f = open("temp.txt", "w")
    batch = " ".join([str(len(idx2word)), "300"])
    for i in range(len(idx2word)):
        line = " ".join([idx2word[i], " ".join([str(num) for num in embedding[i].tolist()])])
        batch = "\n".join([batch, line])
        if len(batch) >= 1000000:
            f.write(batch)
            batch = ""
    f.write(batch)


    print(embedding)