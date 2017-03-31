import os
import tensorflow as tf
import collections

home = os.environ["HOME"]

class Config:
    def __init__(self):
        data_path = "/".join("data")
        self.train_path = os.path.join(data_path, "data.txt")

        self.batch_size = 20
        self.vocab_size = 10000
        self.hidden_units = 200
        self.num_steps = 20
        self.batch_size = 10

class Data:
    def __init__(self, conf):
        self.conf = conf
        self.populate()

    def populate(self):
        self.word_to_id = self.read_words(self.conf.train_path)
        self.train_data = self.file_to_word_ids(self.conf.train_path, self.word_to_id)
        self.train_data = 

    '''
    Helper function
    '''
    def read_words(self, filename):
        with tf.gfile.GFile(filename, "r") as f:
            return [self.transform_word(w) for w in f.read().replace("\n", "<eos>").split()]


    def read_words_by_sections(self, filename):
        #todo
        return None

    def build_vocab(self, filename):
        data = self.read_words(filename)

        counter = collections.Counter(data)
        counter = counter.most_common(self.conf.vocab_size)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))

        return word_to_id

    def file_to_word_ids(self, filename, word_to_id):
      data = self.read_words(filename)
      return [word_to_id[word] for word in data if word in word_to_id]

    def transform_word(self, word):
        if not word[0].isalpha():
            word = word[1:]
        if not word[-1].isalpha():
            word = word[:-1]
        return word


