from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers.core import SpatialDropout1D
from keras.preprocessing import sequence
from keras.utils import np_utils 
from keras.layers import LSTM
from keras.layers import Input, Dense, merge
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Concatenate
import numpy as np

#olsgaard.dk
#for testing

with open("data/main_dictionary_size21.txt", "r") as file:
    main_dict = eval(file.read())

word_pos_pairs = main_dict["word_pos_pairs"]
word_array = [[word[0] for word in sentence] for sentence in word_pos_pairs]
pos_array = [[word[1] for word in sentence] for sentence in word_pos_pairs]


x_train = main_dict["encoded_input"].copy()
y_train = main_dict["encoded_label"].copy()

num_words = len(main_dict["word2index"])
num_dimension = 2


# get length of the longest sentence:
seq_length = max([len(s) for s in x_train])

# find the number of distinct categories
#no_cat = set([ys for sent in y_train for ys in sent])
no_cat =len(main_dict["pos2index"])

# pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=seq_length)
y_train = sequence.pad_sequences(y_train, maxlen=seq_length)

print("X TRAIN SHAPE: ", x_train.shape)
print("Y TRAIN SHAPE: ", y_train.shape)

# convert into a one hot vector
y_train =  np.array([np_utils.to_categorical(seq, no_cat+1) for seq in y_train])

print("Y TRAIN SHAPE to categorical : ", y_train.shape)


# Define model
output_dims = 5
no_targets = 68

input_layer = Input(shape=(seq_length,), dtype='int32')


emb = Embedding(input_dim=num_words, output_dim = output_dims, 
    input_length=num_words, mask_zero=True)(input_layer)

emb2 = SpatialDropout1D(.2)(emb)

# forward LSTM
forward = LSTM(128, return_sequences=True)(emb2)
# backward LSTM
backward = LSTM(128, return_sequences=True, go_backwards=True)(emb2)

common = Concatenate(axis=-1)([forward, backward])
dense = TimeDistributed(Dense(128, activation='tanh'))(common)
out = TimeDistributed(Dense(no_targets, activation='softmax'))(dense)

# Initialize model

model = Model(inputs=[input_layer], outputs=[out])
"""
model.compile(optimizer='adam', loss='msle', metrics=['accuracy'])

# train model
model.fit(x_train, y_train, nb_epoch=5)

result = model.predict(x_train, batch_size=1, verbose=0)
for value in result:
    print(value)

print("EXP  ", y_train)
""" 
