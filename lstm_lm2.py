import os
import tensorflow as tf
import numpy as np

home = os.environ["HOME"]

class Config:
    def __init__(self):
        self.batch_size = 500
        self.sen_len = 50
        self.num_epochs = 100

        self.processor = "/gpu:2"
        os.environ['THEANO_FLAGS'] = 'device=cpu,blas.ldflags=-lblas -lgfortran'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)

        self.word_dim = 300
        self.num_layers = 1
        self.lr = 0.001

        self.path_data = "".join([home,
             "/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/data2.txt"])
        self.path_word = "".join([home,
            "/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/dict2.txt"])
        self.path_output = "".join([home,
            "/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/model.txt"])


class ReadingData:
    def __init__(self, conf):
        self.conf = conf
        self.word2idx = {}
        self.idx2word = []
        f_dic = open(self.conf.path_word, "r")
        for line in f_dic:
            word = line.split("\t")[0]
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)

        self.vocab_size = len(self.idx2word)

    def batch_generator(self):
        while True:
            batch_x = [[0] * self.conf.sen_len] * self.conf.batch_size
            batch_y = [[0] * self.conf.sen_len] * self.conf.batch_size
            batch_idx = 0
            f_doc = open(self.conf.path_data, "r")
            for line in f_doc:
                words = line.split()
                i = 0
                for word in words:
                    if i < self.conf.sen_len:
                        batch_x[batch_idx][i] = word
                    if i + 1 < self.conf.sen_len:
                        batch_y[batch_idx][i+1] = word
                    i+=1
                batch_idx += 1
                if batch_idx == self.conf.batch_size:
                    yield batch_x, batch_y
                    batch_x = [[0] * self.conf.sen_len] * self.conf.batch_size
                    batch_y = [[0] * self.conf.sen_len] * self.conf.batch_size
                    batch_idx = 0
            yield None, None


class ReadingModel:
    def __init__(self, conf, data):
        self.conf = conf
        self.data = data

        import logging
        logging.getLogger().setLevel(logging.INFO)

    def train(self):
        ph_x = tf.placeholder(tf.int32, [None, self.conf.sen_len], name="input_x")
        ph_y = tf.placeholder(tf.int32, [None, self.conf.sen_len], name="input_y")

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self.conf.word_dim, forget_bias=0.0, state_is_tuple=True)
        cell = tf.contrib.rnn.MultiRNNCell(
            [lstm_cell() for _ in range(self.conf.num_layers)], state_is_tuple=True)
        self._initial_state = cell.zero_state(self.conf.batch_size, tf.float32)

        with tf.device(self.conf.processor):
            embedding = tf.get_variable(
                "embedding", [self.data.vocab_size, self.conf.word_dim], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, ph_x)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(self.conf.sen_len):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, self.conf.word_dim])
        softmax_w = tf.get_variable(
            "softmax_w", [self.conf.word_dim, self.data.vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.data.vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(ph_y, [-1])],
            [tf.ones([self.conf.batch_size * self.conf.sen_len], dtype=tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / self.conf.batch_size
        self._final_state = state

        self.train_step = tf.train.GradientDescentOptimizer(self.conf.lr).minimize(self._cost)

        gen = self.data.batch_generator()
        tf.global_variables_initializer().run()
        i = 0
        while True:
            batch_x, batch_y = next(gen)
            if batch_x is None or batch_y is None:
                print("\t".join(["Epoch", str(i), "Finished"]))
                embedding_data = embedding.eval()
                np.savetxt(self.conf.path_output, embedding_data)
                i += 1
                if i == self.conf.num_epochs:
                    break
                else:
                    continue

            self.conf.sess.run(self.train_step, feed_dict={ph_x:batch_x, ph_y:batch_y})




if __name__ == '__main__':
    conf = Config()
    data = ReadingData(conf)
    model = ReadingModel(conf, data)
    model.train()



