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
        self.path = "".join([home,
                             "/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/"])
        self.max_sent_len = 0
        self.stemmer = SnowballStemmer("english")
        self.word2idx = {"<UNK>": 0}
        self.idx2word = ["<UNK>"]
        self.data = []
        self.pre_process()


    def pre_process(self):
        files = listdir(self.path)
        for file in files:
            file = "".join([self.path, file])
            self.count_sentences(file)
        for file in files:
            file = "".join([self.path, file])
            self.process_sent(file)


    def count_sentences(self, path):
        f = open(path, "r")
        for line in f:
            for sent in sent_tokenize(line):
                words = word_tokenize(sent)
                self.max_sent_len = max(len(words), self.max_sent_len)
        np.save("".join([self.path, "data"]), self.data)
        np.save("".join([self.path, "idx2word"]), self.idx2word)
        np.save("".join([self.path, "word2idx"]), self.word2idx)

    def process_sent(self, path):
        f = open(path, "r")
        for line in f:
            for sent in sent_tokenize(line):
                words = word_tokenize(sent)
                data_cur = []
                for word in words:
                    word = self.stemmer.stem(word)
                    if word not in self.word2idx:
                        self.word2idx[word] = len(self.word2idx)
                        self.idx2word.append(word)
                    data_cur.append(self.word2idx[word])
                    while len(data_cur) < self.max_sent_len:
                        data_cur.append(0)
                self.data.append(data_cur)

    def get_data(self):
        words_input_pos = np.zeros((self.batch_size, self.max_sent_len))
        words_input_neg = np.zeros((self.batch_size, self.max_sent_len))
        pseudo_output = np.zeros((self.batch_size, self.max_sent_len, 2))
        batch_idx = 0
        while True:
            random.shuffle(self.data)
            for sent in self.data:
                words_input_pos[batch_idx,] = sent

                for i in range(self.max_sent_len):
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








