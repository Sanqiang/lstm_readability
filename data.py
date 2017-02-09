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
        self.max_sent_len = 0
        self.stemmer = SnowballStemmer("english")
        self.word2idx = {"<UNK>": 0}
        self.idx2word = ["<UNK>"]
        self.data = []
        # self.pre_process()

        self.path = "".join([home, "/data/yelp/review.json"])
        self.path_word = "".join([home, "/data/yelp/processed/word.dict"])
        self.path_doc = "".join([home, "/data/yelp/processed/data.txt"])

    def populate_data(self):
        f = open(self.path_doc, "r")
        data_cur = []
        for line in f:
            words = line.split()
            self.data.append([int(word) for word in words])


    '''
    def pre_process(self):
        files = listdir(self.path)
        for file in files:
            if file[0:4] == "news":
                file = "".join([self.path, file])
                self.count_sentences(file)
        print("process sentence")
        for file in files:
            if file[0:4] == "news":
                file = "".join([self.path, file])
                self.process_sent(file)
                print(file)
        print("process data")
        # np.save("".join([self.path, "data"]), self.data)
        # np.save("".join([self.path, "idx2word"]), self.idx2word)
        # np.save("".join([self.path, "word2idx"]), self.word2idx)
    def count_sentences(self, path):
        f = open(path, "r")
        for line in f:
            for sent in sent_tokenize(line):
                words = word_tokenize(sent)
                self.max_sent_len = max(len(words), self.max_sent_len)
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
                self.data.append(data_cur)
    '''

    def get_data(self):
        words_input_pos = np.zeros((self.batch_size, self.max_sent_len))
        words_input_neg = np.zeros((self.batch_size, self.max_sent_len))
        pseudo_output = np.zeros((self.batch_size, self.max_sent_len, 2))
        batch_idx = 0
        while True:
            random.shuffle(self.data)
            for sent in self.data:
                for i in range(len(sent)):
                    words_input_pos[batch_idx, i] = sent[i]

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

    # deprecated
    def temp_yelp(self):
        import json
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
                    word = self.stemmer.stem(word)
                    if word not in self.word2idx:
                        self.word2idx[word] = len(self.word2idx)
                        self.idx2word.append(word)
                    data_cur = " ".join([data_cur, str(self.word2idx[word])])
            data = "\n".join([data, data_cur])
            if len(data) >= 100000:
                f_doc.write(data)
                data = ""


        f_doc.write(data)
        f_doc.close()
        word_list = ""
        for word in self.idx2word:
            word_list = "\n".join([word_list, word])
        f_word = open(self.path_word, "w")
        f_word.write(word_list)
        f_word.close()

        print(self.max_sent_len) #787

if __name__ == '__main__':
    ddd = DataProvider(0)
    ddd.temp_yelp()

