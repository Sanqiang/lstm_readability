from keras.layers.recurrent import Recurrent, time_distributed_dense
from keras.layers import initializations, activations
from keras.engine import InputSpec
import keras.backend as K

class UnderstandRNN(Recurrent):
    def __init__(self, output_dim, init, inner_activation, timesteps, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_activation = activations.get(inner_activation)

        super(UnderstandRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]
        self.states = [None, None]

        self.W_f = self.inner_init((self.input_dim, self.output_dim),
                                   name='{}_W_f'.format(self.name))
        self.W_c = self.inner_init((self.input_dim, self.output_dim),
                                   name='{}_W_c'.format(self.name))


        self.built = True

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]

        x_i = x[:, :self.output_dim]
        x_f = x[:, self.output_dim: 2 * self.output_dim]

        f = K.sigmoid(K.dot(x_f, self.W_f) + K.dot(h_tm1, self.W_f))
        c = f * c_tm1 +  K.tanh(x_c + K.dot(h_tm1 * B_U[2], self.U_c))

    def preprocess_input(self, x):
        input_shape = self.input_spec[0].shape
        input_dim = input_shape[2]
        timesteps = input_shape[1]
        x_i = time_distributed_dense(x, self.W_i, self.b_i, 0,
                                     input_dim, self.output_dim, timesteps)
        x_f = time_distributed_dense(x, self.W_f, self.b_f, 0,
                                     input_dim, self.output_dim, timesteps)

