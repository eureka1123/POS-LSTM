import numpy as np
import pandas as pd
import re
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

t = open("data/training_data.txt", "r")
#w = open("word_array_columns.txt", "w")
#p = open("pos_array_columns.txt", "w")
main_dictionary = {}
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
    if i < 40000:
        word, pos_tag = line.split()
        word_set.add(word)
        pos_set.add(pos_tag)

        word_array[-1].append(word)
        pos_array[-1].append(pos_tag)

        if word in delimiter:
            word_array.append([])
            pos_array.append([])
"""
with open("data/word_array.txt", "w") as file:
    file.write("\n".join(list(" ".join(word_array[i])  for i in range(len(word_array)))))
"""
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

main_dictionary["word2index"] = word2index
main_dictionary["pos2index"] = pos2index
main_dictionary["index2word"] = index2word
main_dictionary["index2pos"] = index2pos

word_pos_pairs = [list(zip(word_array[x], pos_array[x])) for x in range(len(word_array))]

main_dictionary["word_pos_pairs"] = word_pos_pairs


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
main_dictionary["encoded_input"] = word_array_encoded
main_dictionary["encoded_label"] = pos_array_encoded


with open("data/main_dictionary_size{}.txt".format(len(pos_array_encoded)), "w") as file:
    file.write(str(main_dictionary))


