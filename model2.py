# this model only consider positive example

from keras.models import Model
from keras.layers.recurrent import *
from keras.layers.embeddings import *
from keras.layers import *
from keras.optimizers import *
from data import DataProvider
from keras.callbacks import ModelCheckpoint
import os

tag = "test"
os.environ['THEANO_FLAGS'] = 'device=cpu,blas.ldflags=-lblas -lgfortran'

data = DataProvider(batch_size=500)

embed_dim = 200
vocab_size = 1 + len(data.idx2word)
sen_len = data.max_sent_len
word_embed_data = np.random.rand(vocab_size, embed_dim)

print("vocab size: ", vocab_size)
print("padding sent len: ", data.max_sent_len)

words_input_pos = Input(shape=(sen_len,), dtype="int32", name="words_input_pos")

def get_sim(x):
    return K.batch_dot(x[0], x[1], axes=2)

word_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim, trainable=True, name="word_layer",
                       weights=[word_embed_data])
lstm_layer = LSTM(embed_dim, return_sequences=True, name="lstm_layer", consume_less="gpu", input_length=data.max_sent_len)
sim_layer = Lambda(function=get_sim, name="output_layer", output_shape=(sen_len, sen_len))

words_embed_pos = word_layer(words_input_pos)
lstm_embed = lstm_layer(words_embed_pos)
pos_sim = sim_layer([lstm_embed, words_embed_pos])


def hinge_loss(y_true, y_pred):
    loss = sen_len
    for i in range(sen_len):
        loss -= y_pred[:, i, i]
    loss = K.mean(loss)
    return loss


model = Model(input=[words_input_pos], output=[pos_sim])
model.compile(optimizer=Adam(lr=0.0001), loss=hinge_loss)
print(model.summary())

log_path = "".join([data.path,tag, "log"])

model.fit_generator(generator=data.get_data(include_negative=False), nb_worker=1, pickle_safe=True,
                    nb_epoch=100000, samples_per_epoch=30301028,
                    validation_data=data.get_data(include_negative=False, random_pick=True), nb_val_samples=100,
                    callbacks=[
                        ModelCheckpoint(filepath=log_path, verbose=1, save_best_only=False)
                    ])
