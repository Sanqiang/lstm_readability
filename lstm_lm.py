import os
import tensorflow as tf
import numpy as np
import sys

home = os.environ["HOME"]

class Config:
    def gpu_config(self):
        self.batch_size = 100
        self.processor = "/gpu:2"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)

    def cpu_config(self):
        self.batch_size = 100
        self.processor = "/cpu:0"
        self.sess = tf.InteractiveSession()

    def __init__(self, mode = "GPU"):
        if mode == "GPU":
            self.gpu_config()
        elif mode == "CPU":
            self.cpu_config()
        else:
            raise "config mode error!"

        self.sen_len = 50
        self.num_epochs = 100
        self.num_sen = 324085
        self.max_grad_norm = 5

        self.word_dim = 300
        self.num_layers = 1
        self.lr = 0.001

        self.path_data = "".join([home,
             "/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/data2.txt"])
        self.path_word = "".join([home,
            "/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/dict2.txt"])
        self.path_output = "".join([home,
            "/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/model.txt"])

        self.path_embedding_model = "".join([home,
            "/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/embedding.model"])

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
        # import logging
        # logging.getLogger().setLevel(logging.INFO)

    def evaluate_embedding(self):
        if not os.path.exists(self.conf.path_embedding_model):
            f_model = open(self.conf.path_embedding_model, "w")
            f_model.write(str(self.data.vocab_size))
            f_model.write(" ")
            f_model.write(str(self.conf.word_dim))
            f_model.write("\n")

            vec = np.loadtxt(self.conf.path_output)
            f_dic = open(self.conf.path_word, "r")
            batch = ""
            word_idx = 0
            for line in f_dic:
                word = line.split("\t")[0]
                cur_line = " ".join([word] + [str(val) for val in vec[word_idx].tolist()])
                word_idx += 1
                batch = "\n".join([batch, cur_line])
                if len(batch) > 100000:
                    f_model.write(batch)
                    batch = ""
            f_model.write(batch)

        from gensim.models.word2vec import Word2Vec

        model = Word2Vec.load_word2vec_format(self.conf.path_embedding_model)
        while True:
            word = input("source word:")
            print(model.most_similar(word))



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
        self._cost = tf.reduce_sum(loss) / self.conf.batch_size
        self._final_state = state

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          self.conf.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

        # self.train_step = tf.train.GradientDescentOptimizer(self.conf.lr).minimize(self._cost)

        gen = self.data.batch_generator()
        tf.global_variables_initializer().run()
        idx_epoch = 0
        idx_progress = 0.0
        self.conf.sess.run(self._initial_state)
        fetches = {
            "cost": self.cost,
            "final_state": self.final_state,
            "embedding": self.embedding,
            "eval_op":self._train_op
        }

        while True:
            batch_x, batch_y = next(gen)
            if batch_x is None or batch_y is None:
                print("\t".join(["Epoch", str(idx_epoch), "Finished"]))
                # embedding_data = embedding.eval()
                # np.savetxt(self.conf.path_output, embedding_data)
                idx_epoch += 1
                idx_progress = 0
                if idx_epoch == self.conf.num_epochs:
                    break
                else:
                    continue
            feed_dict = {}
            for i, (c, h) in enumerate(model.initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            vals = self.conf.sess.run(fetches, feed_dict)
            # self.conf.sess.run(self._final_state, feed_dict={ph_x: batch_x, ph_y: batch_y})
            # self.conf.sess.run(self.train_step, feed_dict={ph_x:batch_x, ph_y:batch_y})
            if idx_progress % 100 == 0:
                # cost = self.conf.sess.run(self._cost, feed_dict={ph_x:batch_x, ph_y:batch_y})
                progress = float(idx_progress / self.conf.num_sen)
                sys.stdout.write("\t".join(["Current epoch", str(idx_epoch), "with progress", str(progress), "with cost", str(vals["cost"]), "\n"]))
                sys.stdout.flush()
                np.savetxt(self.conf.path_output, vals["embedding"])
            idx_progress += 1


if __name__ == '__main__':
    conf = Config(mode="GPU")
    data = ReadingData(conf)
    model = ReadingModel(conf, data)
    model.train()
    # model.evaluate_embedding()


