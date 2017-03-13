import time
import os

import numpy as np
import tensorflow as tf

from gensim.models.keyedvectors import KeyedVectors


flags = tf.flags
logging = tf.logging
home = os.environ["HOME"]

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "".join([home, "/data/yelp/processed/data.txt"]),
                    "Where the training/test data is stored.")
flags.DEFINE_string("dict_path", "".join([home, "/data/yelp/processed/word.dict"]),
                    "Where the dictionary is stored.")
flags.DEFINE_string("embedding_path", "".join([home, "/data/yelp/processed/embedding.txt"]),
                    "Where the embedding is stored.")
flags.DEFINE_string("embedding_w2v_path", "".join([home, "/data/yelp/processed/embedding_w2v.txt"]),
                    "Where the embedding w2v is stored.")
flags.DEFINE_string("save_path", "".join([home, "/data/yelp/processed/save"]),
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_epochs", 1000, "number of epochs")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

class DataProvider:
    def __init__(self, config):
        self.conf = config
        self.word2idx = {}
        self.idx2word = []
        f_dic = open(FLAGS.dict_path, "r", encoding="utf-8")
        for line in f_dic:
            word = line.split("\t")[0]
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)

        self.vocab_size = len(self.idx2word)

    def data_producer(self):
        while True:
            batch_x = np.array([[0] * self.conf.num_steps] * self.conf.batch_size)
            batch_y = np.array([[0] * self.conf.num_steps] * self.conf.batch_size)
            batch_idx = 0
            f_doc = open(FLAGS.data_path, "r", encoding="utf-8")
            for line in f_doc:
                words = line.split()
                i = 0
                for word in words:
                    word = int(word)
                    if i < self.conf.num_steps:
                        batch_x[batch_idx][i] = word
                    if i - 1 >= 0 and i < self.conf.num_steps:
                        batch_y[batch_idx][i-1] = word
                    i+=1
                batch_idx += 1
                if batch_idx == self.conf.batch_size:
                    yield batch_x, batch_y
                    batch_x = [[0] * self.conf.num_steps] * self.conf.batch_size
                    batch_y = [[0] * self.conf.num_steps] * self.conf.batch_size
                    batch_idx = 0
            yield None, None

class Model(object):
  """The PTB model."""

  def __init__(self, is_training, config, data_provider):
    self.conf = config
    self.data_provider = data_provider

    batch_size = config.batch_size
    num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = data_provider.vocab_size

    with tf.variable_scope("INPUT"):
      self.batch_input = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name="batch_input")
      self.batch_targets = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name="batch_targets")

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      self.embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type(), trainable=True)
      inputs = tf.nn.embedding_lookup(self.embedding, self.batch_input)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.nn.rnn(cell, inputs,
    #                            initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(outputs, 1), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self.batch_targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def print_out_evaluation(self, target_words):
      f_model = open(FLAGS.embedding_w2v_path, "w", encoding="utf-8")
      f_model.write(str(self.data_provider.vocab_size))
      f_model.write(" ")
      f_model.write(str(self.conf.hidden_size))

      vec = np.loadtxt(FLAGS.embedding_path)
      f_dic = open(FLAGS.dict_path, "r", encoding="utf-8")
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
      f_model.close()
      f_dic.close()

      w2vmodel = KeyedVectors.load_word2vec_format(FLAGS.embedding_w2v_path, binary=False)
      batch = ""
      for target_word in target_words:
        cur_line = "\t\t".join([target_word, str(w2vmodel.most_similar(target_word))])
        batch = "\n".join([batch, cur_line])
      print(batch)
      print("==================")

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  # @property
  # def input(self):
  #   return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 1
  num_steps = 50
  hidden_size = 200
  max_epoch = 100
  max_max_epoch = 50
  keep_prob = 1.0
  lr_decay = 1.0
  batch_size = 1000
  vocab_size = 20000
  epoch_size = 22397781 // batch_size


# class MediumConfig(object):
#   """Medium config."""
#   init_scale = 0.05
#   learning_rate = 1.0
#   max_grad_norm = 5
#   num_layers = 2
#   num_steps = 35
#   hidden_size = 650
#   max_epoch = 6
#   max_max_epoch = 39
#   keep_prob = 0.5
#   lr_decay = 0.8
#   batch_size = 20
#   vocab_size = 10000
#
#
# class LargeConfig(object):
#   """Large config."""
#   init_scale = 0.04
#   learning_rate = 1.0
#   max_grad_norm = 10
#   num_layers = 2
#   num_steps = 35
#   hidden_size = 1500
#   max_epoch = 14
#   max_max_epoch = 55
#   keep_prob = 0.35
#   lr_decay = 1 / 1.15
#   batch_size = 20
#   vocab_size = 10000
#
#
# class TestConfig(object):
#   """Tiny config, for testing."""
#   init_scale = 0.1
#   learning_rate = 1.0
#   max_grad_norm = 1
#   num_layers = 1
#   num_steps = 2
#   hidden_size = 2
#   max_epoch = 1
#   max_max_epoch = 1
#   keep_prob = 1.0
#   lr_decay = 0.5
#   batch_size = 20
#   vocab_size = 10000


def run_epoch(session, model, data_provider, config, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  # state = session.run(model.initial_state)
  gen = data_provider.data_producer()

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  idx_epoch = 0
  cost = 0.0
  while True:
    batch_input, batch_targets = next(gen)
    if batch_input is None or batch_targets is None:
        print("\t".join(["Epoch", str(idx_epoch), "Finished"]))
        idx_epoch += 1
        emb = model.embedding.eval(session=session)
        np.savetxt(FLAGS.embedding_path, emb)
        model.print_out_evaluation(["steak", "seafood", "the"])
        if idx_epoch == FLAGS.num_epochs:
            break
        else:
            continue

    feed_dict = {model.batch_input:batch_input, model.batch_targets:batch_targets}
    state = session.run(model.initial_state)
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)

    cost += vals["cost"]
    # state = vals["final_state"]

    costs += cost
    iters += config.num_steps * config.batch_size

    if verbose and idx_epoch % 1000 == 0:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (idx_epoch * config.batch_size * 1.0 / config.epoch_size, np.exp(costs / iters),
             iters * config.batch_size / (time.time() - start_time)))
    idx_epoch += 1

  return np.exp(costs / iters)

def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  # elif FLAGS.model == "medium":
  #   return MediumConfig()
  # elif FLAGS.model == "large":
  #   return LargeConfig()
  # elif FLAGS.model == "test":
  #   return TestConfig()
  # else:
  #   raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")


  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  data_provider = DataProvider(config)

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = Model(is_training=True, config=config, data_provider=data_provider)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    # with tf.name_scope("Valid"):
    #   valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
    #   with tf.variable_scope("Model", reuse=True, initializer=initializer):
    #     mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
    #   tf.summary.scalar("Validation Loss", mvalid.cost)
    #
    # with tf.name_scope("Test"):
    #   test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
    #   with tf.variable_scope("Model", reuse=True, initializer=initializer):
    #     mtest = PTBModel(is_training=False, config=eval_config,
    #                      input_=test_input)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, data_provider, config, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      #   valid_perplexity = run_epoch(session, mvalid)
      #   print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
      #
      # test_perplexity = run_epoch(session, mtest)
      # print("Test Perplexity: %.3f" % test_perplexity)

      # if FLAGS.save_path:
      #   print("Saving model to %s." % FLAGS.save_path)
      #   sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()