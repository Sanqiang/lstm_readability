# this model consider one positive and negative input data
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


tag = "test_addneg_simple"
os.environ['THEANO_FLAGS'] = 'device=cpu,blas.ldflags=-lblas -lgfortran'


#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

data = DataProvider(batch_size=3000, negative_sampling= True)

embed_dim = 200
vocab_size = len(data.idx2word)
sen_len = data.max_sent_len
word_embed_data = np.random.rand(vocab_size, embed_dim)
#pretraining
home = os.environ["HOME"]
#glove_vector = {}
#glove_path = "".join([home, "/data/glove/glove.twitter.27B.200d.txt"])

#for line in open(glove_path, encoding="utf-8"):
#    item = line.split()
#    glove_vector[item[0].lower()] = [float(val) for val in item[1:]]

#for idx in range(word_embed_data.shape[0]):
#    word = data.idx2word[idx]
#    if word in glove_vector:
#        word_embed_data[idx, :] = glove_vector[word]

print("vocab size: ", vocab_size)
print("padding sent len: ", data.max_sent_len)

words_input_pos = Input(shape=(sen_len,), dtype="int32", name="words_input_pos")
words_input_neg = Input(shape=(sen_len,), dtype="int32", name="words_input_neg")

def get_sim(x):
    return K.batch_dot(x[0], x[1], axes=2)

def merge_sim(x):
    return K.concatenate(x, axis=2)

word_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim, trainable=True, name="word_layer",
                       weights=[word_embed_data])
lstm_layer = LSTM(embed_dim, return_sequences=True, name="lstm_layer", consume_less="gpu", input_length=data.max_sent_len)
sim_layer = Lambda(function=get_sim, name="sim_layer")
merge_layer = Lambda(function=merge_sim, name="output_layer", output_shape=(sen_len, sen_len*2))

words_embed_pos = word_layer(words_input_pos)
words_embed_neg = word_layer(words_input_neg)
lstm_embed = lstm_layer(words_embed_pos)
pos_sim = sim_layer([lstm_embed, words_embed_pos])
neg_sim = sim_layer([lstm_embed, words_embed_neg])
merge_embed = merge_layer([pos_sim, neg_sim])


def hinge_loss(y_true, y_pred):
    loss = sen_len
    for i in range(sen_len):
        loss -= y_pred[:, i, i]
        loss += y_pred[:, i, i+sen_len]
    loss = K.mean(K.maximum(loss, 0.0))
    return loss


model = Model(input=[words_input_pos, words_input_neg], output=[merge_embed])
model.compile(optimizer=Adam(lr=0.0001), loss=hinge_loss)
print(model.summary())

log_path = "/".join([home, "data","model" ,tag])

class my_checker_point(Callback):
    def __init__(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        model_path = "/".join([log_path, "model.txt"])
        np.save(model_path, self.model.get_weights())

model.fit_generator(generator=data.get_data(include_negative=True), nb_worker=4, pickle_safe=True,
                    nb_epoch=100, samples_per_epoch=30301028,
                    # validation_data=data.get_data(include_negative=True, random_pick=True), nb_val_samples=100,
                    callbacks=[
                        ModelCheckpoint(filepath="/".join([log_path, "checker.txt"]), verbose=1, save_best_only=False),
                        my_checker_point(word_layer)
                    ])
