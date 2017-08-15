from keras.layers import Dense, Dropout, Activation, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils 
from keras.layers import LSTM
from keras.layers import Input, Dense, merge
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
import numpy as np

#olsgaard.dk
#for testing
x_train = [[i for i in range(8)], [i for i in range(5)]]
y_train = [[i+1 for i in range(8)], [i+1 for i in range(5)]]

num_words = 22
num_dimension = 2


# get length of the longest sentence:
seq_length = max([len(s) for s in x_train])

# find the number of distinct categories
no_cat = set([ys for sent in y_train for ys in sent])
no_cat =len(no_cat)

# pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=seq_length)
y_train = sequence.pad_sequences(y_train, maxlen=seq_length)

# convert into a one hot vector
y_train =  np.array([np_utils.to_categorical(seq, no_cat+2) for seq in y_train])


# Define model
output_dims = 5
no_targets = 10

input_layer = Input(shape=(seq_length,), dtype='int32')


emb = Embedding(input_dim=num_words, output_dim = output_dims, 
    input_length=seq_length, dropout=0.2, mask_zero=True)(input_layer)

# forward LSTM
forward = LSTM(128, return_sequences=True)(emb)
# backward LSTM
backward = LSTM(128, return_sequences=True, go_backwards=True)(emb)

common = merge([forward, backward], mode='concat', concat_axis=-1)
dense = TimeDistributed(Dense(128, activation='tanh'))(common)
out = TimeDistributed(Dense(no_targets, activation='softmax'))(dense)

# Initialize model

model = Model(input=input_layer, output=out)
model.compile(optimizer='adam', loss='msle', metrics=['accuracy'])

# train model
model.fit(x_train, y_train, nb_epoch=300)

result = model.predict(x_train, batch_size=1, verbose=0)
for value in result:
    print(value)

print("EXP  ", y_train) 
