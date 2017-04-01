from data import Data, Config
from rnn import RNN

cfg = Config()
d = Data(conf=cfg)
rnn = RNN(data=d, config=cfg)
rnn.train("test")

print("x")
