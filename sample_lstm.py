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

def array_to_one_hot(list_given):
	max_index = list_given.index(max(list_given))
	x = np.zeros((len(list_given),))
	x[max_index] = 1
	return x

def are_equal(list_1, list_2):
	assert(len(list_1) == len(list_2))
	for i in range(len(list_1)):
		if list_1[i] != list_2[i]:
			return False

	return True


with open("data/main_dictionary_size45.txt", "r") as file:
    main_dict = eval(file.read())

word_pos_pairs = main_dict["word_pos_pairs"]
word_array = [[word[0] for word in sentence] for sentence in word_pos_pairs]
pos_array = [[word[1] for word in sentence] for sentence in word_pos_pairs]


x_train = main_dict["encoded_input"][:40].copy()
y_train = main_dict["encoded_label"][:40].copy()

x_test = main_dict["encoded_input"][40:].copy()
y_test = main_dict["encoded_label"][40:].copy()

num_words = len(main_dict["word2index"])
num_dimension = 2


# get length of the longest sentence:
seq_length = max([len(s) for s in x_train])

# find the number of distinct categories
#no_cat = set([ys for sent in y_train for ys in sent])
no_cat =len(main_dict["pos2index"])

print("XTETS ", x_test)

# pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=seq_length)
y_train = sequence.pad_sequences(y_train, maxlen=seq_length)

x_test = sequence.pad_sequences(x_test, maxlen=seq_length)
y_test = sequence.pad_sequences(y_test, maxlen=seq_length)

print("X TRAIN SHAPE: ", x_train.shape)
print("Y TRAIN SHAPE: ", y_train.shape)

# convert into a one hot vector
y_train =  np.array([np_utils.to_categorical(seq, no_cat+1) for seq in y_train])

y_test =  np.array([np_utils.to_categorical(seq, no_cat+1) for seq in y_test])

#print("Y TRAIN SHAPE to categorical : ", y_train.shape)


# Define model
output_dims = 5
no_targets = y_train.shape[2]

input_layer = Input(shape=(seq_length,), dtype='int32')


emb = Embedding(input_dim=num_words+1, output_dim = output_dims, 
    input_length=seq_length, mask_zero=True)(input_layer)

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

model.compile(optimizer='adam', loss='msle', metrics=['accuracy'])

# train model
model.fit(x_train, y_train, nb_epoch=20)

result = model.predict(x_test, batch_size=1, verbose=0)

result = [array_to_one_hot(list(lst)) for lst in result[0]]

count = 0
right = 0
for i in range(len(result)):
	count += 1
	if are_equal(result[i],y_test[0][i]):
		right += 1

print("Accuracy = ", right/float(count))


