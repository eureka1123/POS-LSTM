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
pos2stpos = {'f':'other','m1':'cd','j':'other','npx':'nnp','zz': 'other',"m":"cd","x":"other","z":"prp$", "z'":"prp$",'vvd@': 'vbd','y': 'other','jj@':'rb','appge': 'prp$', 'at': 'dt', 'at1': 'dt', 'bcl': 'in', 'cc': 'cc', 'ccb': 'cc', 'cs': 'in', 'csa': 'cc', 'csn': 'cc', 'cst': 'cc', 'csw': 'cc', 'da': 'dt', 'da1': 'dt', 'da2': 'dt', 'dar': 'dt', 'dat': 'dt', 'db': 'pdt', 'db2': 'pdt', 'dd': 'dt', 'dd1': 'dt', 'dd2': 'dt', 'ddq': 'wdt', 'ddqge': 'wdt', 'ddqv': 'wdt', 'ex': 'ex', 'fo': 'sym', 'fu': 'other', 'fw': 'fw', 'ge': 'other', 'if': 'in', 'ii': 'in', 'io': 'in', 'iw': 'in', 'jj': 'jj', 'jjr': 'jjr', 'jjt': 'jjs', 'jk': 'jj', 'mc': 'cd', 'mc1': 'cd', 'mc2': 'cd', 'mcge': 'cd', 'mcmc': 'cd', 'md': 'cd', 'mf': 'cd', 'nd1': 'nn', 'nn': 'nn', 'nn1': 'nn', 'nn2': 'nns', 'nna': 'nn', 'nnb': 'nn', 'nnl1': 'nn', 'nnl2': 'nns', 'nno': 'nn', 'nno2': 'nns', 'nnt1': 'nn', 'nnt2': 'nns', 'nnu': 'nn', 'nnu1': 'nn', 'nnu2': 'nns', 'np': 'nnp', 'np1': 'nnp', 'np2': 'nnps', 'npd1': 'nnp', 'npd2': 'nnps', 'npm1': 'nnp', 'npm2': 'nnps', 'pn': 'prp', 'pn1': 'prp', 'pnqo': 'wp', 'pnqs': 'wp', 'pnqv': 'wp', 'pnx1': 'prp', 'ppge': 'prp$', 'pph1': 'prp', 'ppho1': 'prp', 'ppho2': 'prp', 'pphs1': 'prp', 'pphs2': 'prp', 'ppio1': 'prp', 'ppio2': 'prp', 'ppis1': 'prp', 'ppis2': 'prp', 'ppx1': 'prp', 'ppx2': 'prp', 'ppy': 'prp', 'ra': 'rb', 'rex': 'rb', 'rg': 'rb', 'rgq': 'wrb', 'rgqv': 'wrb', 'rgr': 'rbr', 'rgt': 'rbs', 'rl': 'rb', 'rp': 'rb', 'rpk': 'rb', 'rr': 'rb', 'rrq': 'wrb', 'rrqv': 'wrb', 'rrr': 'rbr', 'rrt': 'rbs', 'rt': 'rb', 'to': 'to', 'uh': 'uh', 'vb0': 'vb', 'vbdr': 'vbd', 'vbdz': 'vbd', 'vbg': 'vbg', 'vbi': 'vb', 'vbm': 'vbp', 'vbn': 'vbn', 'vbr': 'vbr', 'vbz': 'vbz', 'vd0': 'vb', 'vdd': 'vbd', 'vdg': 'vbg', 'vdi': 'vb', 'vdn': 'vbn', 'vdz': 'vbz', 'vh0': 'vb', 'vhd': 'vbd', 'vhg': 'vbg', 'vhi': 'vb', 'vhn': 'vbn', 'vhz': 'vbz', 'vm': 'vb', 'vmk': 'vb', 'vv0': 'vb', 'vvd': 'vbd', 'vvg': 'vbg', 'vvgk': 'vbg', 'vvi': 'vb', 'vvn': 'vbn', 'vvnk': 'vbn', 'vvz': 'vbz', 'xx': 'rb', 'zz1': 'other', 'zz2': 'other'}

delimiter = set([".","?","!" ])
for i, line in enumerate(t):
    if i < 300000:
        word, pos_tag = line.split()
        word_set.add(word)
	
        
        pos_tag = pos2stpos[pos_tag]
        
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


with open("data/main_dictionary_size_new_300K_{}.txt".format(len(pos_array_encoded)), "w") as file:
    file.write(str(main_dictionary))


