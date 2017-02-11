from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
import os
from os import listdir
import numpy as np
import random

home = os.environ["HOME"]

class DataProvider:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        #self.path = "".join([home,
        #                     "/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/"])
        self.max_sent_len = 50
        self.stemmer = SnowballStemmer("english")
        self.word2idx = {"<UNK>": 0}
        self.idx2word = ["<UNK>"]
        self.data = []

        self.path = "".join([home, "/data/yelp/review.json"])
        self.path_word = "".join([home, "/data/yelp/processed/word.dict"])
        self.path_doc = "".join([home, "/data/yelp/processed/data.txt"])
        self.path_doc2 = "".join([home, "/data/yelp/processed/data2.txt"])

        # self.populate_dict()
        # self.populate_data()

    def populate_dict(self):
        f = open(self.path_word, "r")
        for line in f:
            word = line.strip()
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)

    def populate_data(self):
        f = open(self.path_doc, "r")
        for line in f:
            words = line.split()
            self.data.append([int(word) for word in words])

    def get_data(self):
        words_input_pos = np.zeros((self.batch_size, self.max_sent_len))
        words_input_neg = np.zeros((self.batch_size, self.max_sent_len))
        pseudo_output = np.zeros((self.batch_size, self.max_sent_len, 2))
        batch_idx = 0
        while True:
            random.shuffle(self.data)
            for sent in self.data:
                for i in range(len(sent)):
                    if i >= self.max_sent_len:
                        break
                    words_input_pos[batch_idx, i] = sent[i]

                for i in range(self.max_sent_len):
                    if i >= self.max_sent_len:
                        break
                    word = random.randint(1, len(self.idx2word))
                    while word in sent:
                        word = random.randint(1, len(self.idx2word))
                    words_input_neg[batch_idx, i] = word

                batch_idx += 1

                if batch_idx == self.batch_size:
                    yield ({"words_input_pos": words_input_pos, "words_input_neg": words_input_neg},
                           {"merge_layer": pseudo_output})
                    words_input_pos = np.zeros((self.batch_size, self.max_sent_len))
                    words_input_neg = np.zeros((self.batch_size, self.max_sent_len))
                    batch_idx = 0

    def word_transform(self, word):
        word = word.lower()
        word = self.stemmer.stem(word)

        for i in range(len(word)):
            if word[i] < 'a' or word[i] > 'z':
                return "<UNK>"
        return word

    # deprecated
    def temp_yelp(self):
        import json
        word2idx = {"<UNK>": 0}
        idx2word =["<UNK>"]

        f = open(self.path, "r")
        f_doc = open(self.path_doc, "w")
        data = ""
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            for sent in sent_tokenize(text):
                data_cur = ""
                words = word_tokenize(sent)
                self.max_sent_len = max(self.max_sent_len, len(words))
                for word in words:
                    word = self.word_transform(word)
                    if word not in word2idx:
                        word2idx[word] = len(word2idx)
                        idx2word.append(word)
                    data_cur = " ".join([data_cur, str(word2idx[word])])
            data = "\n".join([data, data_cur])
            if len(data) >= 100000:
                f_doc.write(data)
                data = ""
        f_doc.write(data)
        f_doc.close()


        print(self.max_sent_len) #787

        f_doc = open(self.path_doc, "r")
        word_cnt = {}
        for line in f_doc:
            words = line.split()
            for word in words:
                word_cnt[word] += 1
        f_doc = open(self.path_doc, "r")
        data = ""
        f_doc2 = open(self.path_doc2, "w")
        nword2idx = {"<UNK>":0}
        nidx2word = ["<UNK>"]
        for line in f_doc:
            data_cur = ""
            words = line.split()
            for word in words:
                if word_cnt[word] >= 3:
                    data_cur = " ".join([data_cur, word])
                    nword2idx[idx2word[word]] = len(nword2idx)
                    nidx2word.append(idx2word[word])

            data = "\n".join([data, data_cur])
            if len(data) >= 100000:
                f_doc2.write(data)
                data = ""
        f_doc2.write(data)
        f_doc2.close()

        word_list = ""
        for word in nidx2word:
            word_list = "\n".join([word_list, word])
        f_word = open(self.path_word, "w")
        f_word.write(word_list)
        f_word.close()


if __name__ == '__main__':
    ddd = DataProvider(0)
    ddd.temp_yelp()

