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
word_set = set()
pos_set = set()
word_array = []
pos_array = []
case = []
for i, line in enumerate(t):
    if i < 5:
        print(i,line)
        word, pos_tag = line.split()
        word_set.add(word)
        pos_set.add(pos_tag)

        word_array.append(word)
        pos_array.append(pos_tag)

count = 0
for word in word_set:
    count += 1
    word2index[word] = count

count = 0
for tag in pos_set:
    count += 1
    pos2index[tag] = count

with open("word2index.txt", "w") as file:
    file.write(str(word2index))

with open("pos2index.txt", "w") as file:
    file.write(str(pos2index))

t.close()
