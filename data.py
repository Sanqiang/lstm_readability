import os
import collections
import tensorflow as tf
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import pickle


home = os.environ["HOME"]

class Config:
    def __init__(self):
        data_path = "/".join([home, "data/yelp"])
        self.train_path = os.path.join(data_path, "review.json")

        self.batch_size = 1
        self.vocab_size = 10000
        self.hidden_units = 200
        self.num_steps = 10
        self.batch_size = 20
        self.max_gradient = 5.0
        self.init_random = .6

class Data:


    def __init__(self, conf):
        self.conf = conf
        self.populate()

    def populate(self):
        if os.path.exists("pickle"):
            self.word2idx, self.idx2words, self.xs, self.ys = pickle.load(open("pickle", "rb"))
        else:
            self.word2idx, self.idx2words, self.xs, self.ys = self.build_vocab()
            pickle.dump([self.word2idx, self.idx2words, self.xs, self.ys], open("pickle", "wb"))


        '''
        self.word_to_id = self.build_vocab(self.conf.train_path)
        self.train_data = self.file_to_word_ids(self.conf.train_path, self.word_to_id)
        # for train
        n = len(self.train_data)
        m = self.conf.num_steps * self.conf.batch_size
        p = [(n + i) % m for i in range(m)].index(0)
        self.padding_train_data = np.pad(self.train_data, (0, (p + 1)), mode="constant")
        instances = np.math.ceil((len(self.padding_train_data) - 1) / self.conf.num_steps)
        # Context and targets are arrays of shape (total_batches x batch_size x time_steps).
        xs = self.padding_train_data[:-1].reshape(instances // self.conf.batch_size, self.conf.batch_size, self.conf.num_steps)
        ys = self.padding_train_data[1:].reshape(instances // self.conf.batch_size, self.conf.batch_size, self.conf.num_steps)
        self.pair_padding_train_data = zip(xs, ys)
        '''

    def get_data(self):
        xs = np.zeros((self.conf.batch_size, self.conf.num_steps))
        ys = np.zeros((self.conf.batch_size, self.conf.num_steps))
        batch_idx = 0
        while True:
            for x, y in zip(self.xs, self.ys):
                x = np.pad(x, (0, max(0, self.conf.num_steps - len(x))), "constant")
                y = np.pad(y, (0, max(0, self.conf.num_steps - len(y))), "constant")
                xs[batch_idx] = x[:self.conf.num_steps]
                ys[batch_idx] = y[:self.conf.num_steps]
                batch_idx += 1
                if len(xs) == self.conf.batch_size:
                    yield xs, ys
                    xs = np.zeros((self.conf.batch_size, self.conf.num_steps))
                    ys = np.zeros((self.conf.batch_size, self.conf.num_steps))
                    batch_idx = 0

    '''
    Helper function
    '''
    def build_vocab(self):
        counter = collections.Counter()
        f = open(self.conf.train_path, encoding="utf-8")
        nsents = []
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            for sent in sent_tokenize(text):
                nsent = []
                for word in word_tokenize(sent):
                    word = self.transform_word(word)
                    nsent.append(word)
                counter.update(nsent)
                nsents.append(nsent)

        counter = counter.most_common(self.conf.vocab_size)
        idx2words = list(zip(*counter))[0]
        word2idx = dict(zip(idx2words, range(len(idx2words))))

        xs = []
        ys = []
        for nsent in nsents:
            nsent_filter = []
            for word in nsent:
                if word in word2idx:
                    nsent_filter.append(word2idx[word])

            xs.append(nsent_filter[:-1])
            ys.append(nsent_filter[1:])
            # xs.append(np.pad(nsent_filter[:-1],(0, max(0, self.conf.num_steps-len(nsent_filter[:-1]))), 'constant'))
            # ys.append(np.pad(nsent_filter[1:],(0,max(0, self.conf.num_steps-len(nsent_filter[1:]))), 'constant'))

        del nsents

        return word2idx, idx2words, xs, ys

    def transform_word(self, word):
        if len(word) > 0:
            if not word[0].isalpha():
                word = word[1:]
            if len(word) > 0 and not word[-1].isalpha():
                word = word[:-1]
        return word

    '''
        deprecated

    def read_words(self, filename):
        with tf.gfile.GFile(filename, "r") as f:
            return [w for w in (f.read().decode("utf-8") .replace("\n", "<eos>").split())]
            # return f.read().decode("utf-8") .replace("\n", "<eos>").split()


    def read_words_by_sections(self, filename):
        #todo
        return None

    def build_vocab(self, filename):
        data = self.read_words(filename)

        counter = collections.Counter(data)
        count_pairs = counter.most_common(self.conf.vocab_size)
        # count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))

        return word_to_id

    def file_to_word_ids(self, filename, word_to_id):
      data = self.read_words(filename)
      return [word_to_id[word] for word in data if word in word_to_id]



'''
