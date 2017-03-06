# this model temp
from keras.models import Model
from keras.layers.recurrent import *
from keras.layers.embeddings import *
from keras.layers import *
from keras.optimizers import *
from data import DataProvider
from keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf
from keras.callbacks import Callback

sen_len = 4
vocab_size = 10
embed_dim = 3

word_embed = np.zeros((vocab_size, embed_dim), dtype=np.float)
word_embed[1,0] = 1
word_embed[1,1] = 2
word_embed[1,2] = 3
word_embed[2,0] = 0.5
word_embed[2,1] = 0.5
word_embed[2,2] = -1

if True:
    # test
    def get_diag(x):
        return x[:,0:2,0:2]

    inputx = Input(shape=(sen_len,), dtype="int32", name="inputx")
    inputy = Input(shape=(sen_len,), dtype="int32", name="inputy")
    word_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim, trainable=True, name="word_layer", weights=[word_embed], mask_zero=True)
    sim_layer = Merge(mode="dot", dot_axes=2, name="sim_layer")
    diag_layer = Lambda(function=get_diag, name="diag_layer")

    w_inputx = word_layer(inputx)
    w_inputy = word_layer(inputy)

    dot_output = sim_layer([w_inputx, w_inputy])
    diag_output = diag_layer(dot_output)
    model = Model(input=[inputx, inputy], output=[diag_output])
    print(model.summary())

    result = model.predict({"inputx":np.array([[1,2,0,0]]), "inputy":np.array([[2,2,0,0]])}, batch_size=1)
    print(result)

if False:
    words_input_pos = Input(shape=(sen_len,), dtype="int32", name="words_input_pos")
    words_input_neg = Input(shape=(sen_len,), dtype="int32", name="words_input_neg")
    word_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim, trainable=True, name="word_layer")

    def merge_sim(x):
        return K.concatenate(x, axis=2)

    word_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim, trainable=True, name="word_layer")
    lstm_layer = LSTM(embed_dim, return_sequences=True, name="lstm_layer", consume_less="gpu", input_length=sen_len)
    sim_layer = Merge(mode="dot", dot_axes=2, name="sim_layer")
    # merge_layer = Lambda(function=merge_sim, name="output_layer", output_shape=(sen_len, sen_len*2))
    mask_layer = Masking(mask_value=0)

    words_embed_pos = word_layer(words_input_pos)
    # words_embed_neg = word_layer(words_input_neg)
    words_embed_pos = mask_layer(words_embed_pos)
    # words_embed_neg = mask_layer(words_embed_neg)
    lstm_embed = lstm_layer(words_embed_pos)
    pos_sim = sim_layer([lstm_embed, words_embed_pos])
    # neg_sim = sim_layer([lstm_embed, words_embed_neg])
    # merge_embed = merge_layer([pos_sim, neg_sim])

    model = Model(input=[words_input_pos, words_input_neg], output=[pos_sim])

    print(model.summary())

    result = model.predict({}, batch_size=1)

    print(result)
