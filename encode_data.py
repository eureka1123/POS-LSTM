import numpy as np
import pandas as pd
import re
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

t = open("training_data.txt", "r")
#w = open("word_array_columns.txt", "w")
#p = open("pos_array_columns.txt", "w")

word2index = {}
pos2index = {}
index2word = {}
index2pos = {}
word_set = set()
pos_set = set()
word_array = [[]]
pos_array = [[]]
case = []
delimiter = set([".","?","!" ])
for i, line in enumerate(t):
    if i < 500:
        word, pos_tag = line.split()
        word_set.add(word)
        pos_set.add(pos_tag)

        word_array[-1].append(word)
        pos_array[-1].append(pos_tag)

        if word in delimiter:
            word_array.append([])
            pos_array.append([])

count = 0
for word in word_set:
    count += 1
    word2index[word] = count
    index2word[count] = word

count = 0
for tag in pos_set:
    count += 1
    pos2index[tag] = count
    index2pos[count] = tag

with open("word2index.txt", "w") as file:
    file.write(str(word2index))

with open("pos2index.txt", "w") as file:
    file.write(str(pos2index))


with open("index2word.txt", "w") as file:
    file.write(str(index2word))

with open("index2pos.txt", "w") as file:
    file.write(str(index2pos))


t.close()
"""
print("WORD Array ", word_array)
print("POS Array ", pos_array)
"""
word_array_encoded = [[word2index.get(w) for w in s] for s in word_array]
pos_array_encoded = [[pos2index.get(w) for w in s] for s in pos_array]
"""
print("WORD ARRAY ENCODED ", list(word_array_encoded))
print("POS Array ENCODED ", list(pos_array_encoded))
"""

with open("encoded_input_size{}.txt".format(len(word_array_encoded)), "w") as file:
    file.write(str(word_array_encoded))

with open("encoded_label_size{}.txt".format(len(pos_array_encoded)), "w") as file:
    file.write(str(pos_array_encoded))

