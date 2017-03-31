import tensorflow as tf

class RNN:
    def __init__(self, vocabulary_size, hidden_units, num_steps, batch_size, max_gradient):
        with tf.name_scope("HyperParameter"):
            self.learning_rate  = tf.placeholder(tf.float32, name="learning_rate")


        with tf.name_scope("Inputs"):
            self.inputs = tf.placeholder(tf.int32, name="inputs", shape=(batch_size, num_steps))
            self.targets = tf.placeholder(tf.int32, name="targets", shape=(batch_size, num_steps))
            self.init_random = tf.placeholder(tf.float32, name="init")

        with tf.name_scope("Embedding"):
            self.embedding = tf.Variable(tf.random_uniform((vocabulary_size, hidden_units), -self.init_random, self.init_random),
                                         dtype=tf.float32,
                                         name="embedding")
            self.embedded_input = tf.nn.embedding_lookup(self.embedding, self.inputs, name="embedded_input")

        with tf.name_scope("RNN"):
            def lstm_cell():
                return tf.contrib.rnn.BasicLSTMCell(hidden_units, forget_bias = 0.0, state_is_tuple = True)
            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_steps)], state_is_tuple = True)
            self.reset_state = cell.zero_state(batch_size, tf.float32)
            state = self.reset_state
            outputs = []
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(self.inputs[:, time_step, :], state)
                outputs.append(cell_output)
            self.outputs = tf.reshape(tf.concat(1, outputs), [-1, hidden_units])

        with tf.name_scope("Cost"):
            self.flattened_outputs = tf.reshape(tf.concat(1, self.outputs), (-1, hidden_units),
                                                name="flattened_outputs")

            softmax_w = tf.get_variable("softmax_w", [hidden_units, vocabulary_size])
            softmax_b = tf.get_variable("softmax_b", [vocabulary_size])
            logits = tf.matmul(self.flattened_outputs, softmax_w) + softmax_b
            loss = tf.nn.seq2seq.sequence_loss_by_example(
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
        self.summary = tf.merge_all_summaries()


    def train(self, data, init, save_path):
        sv = tf.train.Supervisor(logdir=save_path)
        with sv.managed_session() as session:
            session.run(self.initialize, feed_dict={self.init_random: init})
            for pair in data.pair_padding_train_data():
                if True: #start document
                    state = session.run(self.reset_state)
                _, cost, state, iteration = session.run(
                    [self.train_step, self.cost, self.next_state, self.iteration],
                    feed_dict={
                        self.input: pair[0],
                        self.targets: pair[1],
                        self.state: state,
                        self.learning_rate: 0.001,
                        })
        print("Saving model to %s." % save_path)
        sv.saver.save(session, save_path, global_step=sv.global_step)

        return None
