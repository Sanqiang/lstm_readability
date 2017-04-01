import tensorflow as tf
import inspect

class RNN:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.model(vocabulary_size=config.vocab_size,
                   hidden_units=config.hidden_units,
                   num_steps=config.num_steps,
                   batch_size=config.batch_size,
                   max_gradient=config.max_gradient)


    def model(self, vocabulary_size, hidden_units, num_steps, batch_size, max_gradient):
        print("for tf", str(tf.__version__))

        with tf.name_scope("HyperParameter"):
            self.learning_rate  = tf.placeholder(tf.float32, name="learning_rate")

        with tf.name_scope("Inputs"):
            self.inputs = tf.placeholder(tf.int32, name="inputs", shape=(batch_size, num_steps))
            self.targets = tf.placeholder(tf.int32, name="targets", shape=(batch_size, num_steps))
            self.init_random = tf.placeholder(tf.float32, name="init")

        with tf.name_scope("Embedding"):
            self.embedding = tf.get_variable("embedding", [vocabulary_size, hidden_units])
            softmax_w = tf.get_variable("softmax_w", [hidden_units, vocabulary_size])
            softmax_b = tf.get_variable("softmax_b", [vocabulary_size])
            self.embedded_input = tf.nn.embedding_lookup(self.embedding, self.inputs, name="embedded_input")

        with tf.name_scope("RNN"):
            def lstm_cell():
                if 'reuse' in inspect.getargspec(
                        tf.contrib.rnn.BasicLSTMCell.__init__).args:
                    return tf.contrib.rnn.BasicLSTMCell(
                        hidden_units, forget_bias=0.0, state_is_tuple=True,
                        reuse=tf.get_variable_scope().reuse)
                else:
                    return tf.contrib.rnn.BasicLSTMCell(
                        hidden_units, forget_bias=0.0, state_is_tuple=True)
            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell()] * 1)
            self.initial_state = cell.zero_state(batch_size, tf.float32)
            state = self.initial_state
            outputs = []
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(self.embedded_input[:, time_step], state)
                outputs.append(cell_output)
            self.outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_units])

        with tf.name_scope("Cost"):
            self.flattened_outputs = tf.reshape(tf.concat(axis=1, values=self.outputs), (-1, hidden_units),
                                                name="flattened_outputs")


            logits = tf.matmul(self.flattened_outputs, softmax_w) + softmax_b
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([batch_size * num_steps])])
            self.cost = tf.reduce_sum(loss) / batch_size
            self.final_state = state

        with tf.name_scope("Train"):
            self.iteration = tf.Variable(0, dtype=tf.int64, name="iteration", trainable=False)
            self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()),
                                                       max_gradient, name="clip_gradients")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            self.train_step = optimizer.apply_gradients(zip(self.gradients, tf.trainable_variables()),
                                                        name="train_step",
                                                        global_step=self.iteration)

        self.initialize = tf.initialize_all_variables()


    def train(self, save_path):
        sv = tf.train.Supervisor(logdir=save_path)
        with sv.managed_session() as session:
            state = session.run(self.initial_state)

            session.run(self.initialize, feed_dict={self.init_random: self.config.init_random})
            for pair in self.data.pair_padding_train_data:

                _, cost, state, iteration = session.run(
                    [self.train_step, self.cost, self.final_state, self.iteration],
                    feed_dict={
                        self.inputs: pair[0],
                        self.targets: pair[1],
                        self.initial_state: state,
                        self.learning_rate: 0.001,
                    })
            print("Saving model to %s." % save_path)
            sv.saver.save(session, save_path, global_step=sv.global_step)
