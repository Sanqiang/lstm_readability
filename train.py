from data import Config, Data
from rnn import RNN

config = Config()
data = Data(config)
rnn = RNN(vocabulary_size=config.vocab_size, hidden_units=200, max_gradient = 5.0,
          num_steps=config.num_steps, batch_size=config.batch_size)



