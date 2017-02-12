from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
import os
from os import listdir
import numpy as np
import random
from collections import Counter

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

        # self.path = "".join([home, "/data/yelp/review.json"])
        # self.path_word = "".join([home, "/data/yelp/processed/word.dict"])
        # self.path_doc = "".join([home, "/data/yelp/processed/data.txt"])
        # self.path_doc2 = "".join([home, "/data/yelp/processed/data2.txt"])
        #
        self.path_news = "".join([home, "/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled"])
        self.path_news_doc = "".join([home,
             "/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/data2.txt"])
        self.path_news_dict = "".join([home,
            "/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/dict2.txt"])

        self.path = self.path_news
        self.path_word = self.path_news_dict
        self.path_doc = self.path_news_doc

        self.populate_dict()
        self.populate_data()

    def populate_dict(self):
        f = open(self.path_word, "r")
        for line in f:
            words = line.split()
            for word in words:
                word = line.strip()
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)

    def populate_data(self):
        f = open(self.path_doc, "r")
        for line in f:
            words = line.split()
            self.data.append([int(word) for word in words])

    def get_data(self, include_negative):
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

                if include_negative:
                    for i in range(self.max_sent_len):
                        if i >= self.max_sent_len:
                            break
                        word = random.randint(1, len(self.idx2word))
                        while word in sent:
                            word = random.randint(1, len(self.idx2word))
                        words_input_neg[batch_idx, i] = word

                batch_idx += 1

                if batch_idx == self.batch_size:
                    if include_negative:
                        yield ({"words_input_pos": words_input_pos, "words_input_neg": words_input_neg},
                               {"output_layer": pseudo_output})
                    else:
                        yield ({"words_input_pos": words_input_pos},
                               {"output_layer": pseudo_output})
                    words_input_pos = np.zeros((self.batch_size, self.max_sent_len))
                    words_input_neg = np.zeros((self.batch_size, self.max_sent_len))
                    batch_idx = 0

    def word_transform(self, word):
        word = word.lower()
        # word = self.stemmer.stem(word)
        #
        # for i in range(len(word)):
        #     if word[i] < 'a' or word[i] > 'z':
        #         return "<UNK>"
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
                wordidxs = word_tokenize(sent)
                self.max_sent_len = max(self.max_sent_len, len(wordidxs))
                for wordidx in wordidxs:
                    wordidx = self.word_transform(wordidx)
                    if wordidx not in word2idx:
                        word2idx[wordidx] = len(word2idx)
                        idx2word.append(wordidx)
                    data_cur = " ".join([data_cur, str(word2idx[wordidx])])
            data = "\n".join([data, data_cur])
            if len(data) >= 100000:
                f_doc.write(data)
                data = ""
        f_doc.write(data)
        f_doc.close()


        print(self.max_sent_len) #787


        # refine
        f_doc = open(self.path_doc, "r")
        wordidx_cnt = {}
        for line in f_doc:
            wordidxs = line.split()
            for wordidx in wordidxs:
                if wordidx not in wordidx_cnt:
                    wordidx_cnt[wordidx] = 0
                wordidx_cnt[wordidx] += 1
        f_doc = open(self.path_doc, "r")
        data = ""
        f_doc2 = open(self.path_doc2, "w")
        nword2idx = {"<UNK>":0}
        nidx2word = ["<UNK>"]
        for line in f_doc:
            data_cur = ""
            wordidxs = line.split()
            for wordidx in wordidxs:
                if wordidx_cnt[wordidx] >= 3:
                    if wordidx not in nword2idx:
                        nword2idx[idx2word[int(wordidx)]] = len(nword2idx)
                        nidx2word.append(idx2word[int(wordidx)])
                    nwordidx = nword2idx[idx2word[int(wordidx)]]
                    data_cur = " ".join([data_cur, str(nwordidx)])


            data = "\n".join([data, data_cur])
            if len(data) >= 100000:
                f_doc2.write(data)
                data = ""
        f_doc2.write(data)
        f_doc2.close()

        word_list = ""
        for wordidx in nidx2word:
            word_list = "\n".join([word_list, wordidx])
        f_word = open(self.path_word, "w")
        f_word.write(word_list)
        f_word.close()

    def temp_news(self):
        word2cnt = Counter()

        files = os.listdir(self.path_news)
        for file in files:
            if file[0:4] != "news":
                continue
            file = "/".join([self.path_news, file])
            for line in open(file, "r"):
                words = line.split()
                for word in words:
                    word = self.word_transform(word)
                    if word not in word2cnt:
                        word2cnt[word] = 0
                    word2cnt[word] += 1
            print("finished", file)
        print("finish word count", str(len(word2cnt)))
        nword2cnt = {}
        for word in word2cnt:
            if word2cnt[word] > 5:
                nword2cnt[word] = word2cnt[word]
        word2cnt = nword2cnt
        print("current word count", str(len(word2cnt)))
        f = open(self.path_news_doc, "w")
        data = ""
        for file in files:
            if file[0:4] != "news":
                continue
            file = "/".join([self.path_news, file])
            for line in open(file, "r"):
                cur_line = ""
                words = line.split()
                for word in words:
                    word = self.word_transform(word)
                    if word in word2cnt:
                        if word not in self.word2idx:
                            self.word2idx[word] = len(self.word2idx)
                            self.idx2word.append(word)
                    else:
                        word = "<UNK>"
                    cur_line = " ".join((cur_line, str(self.word2idx[word])))
                data = "\n".join((data, cur_line))
                if len(data) > 100000:
                    f.write(data)
                    data = ""
            print("finished", file)
        f.write(data)
        f.close()

        f = open(self.path_news_dict, "w")
        for word in self.idx2word:
            f.write(word)
            f.write("\n")
        f.close()


if __name__ == '__main__':
    ddd = DataProvider(0)
    # ddd.temp_yelp()
    ddd.temp_news()
