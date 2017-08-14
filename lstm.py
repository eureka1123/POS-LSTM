import numpy as np 
import pandas as pd 
import re 

t = open("training_data.txt", "r")
w = open("word_array_columns.txt", "w")
p = open("pos_array_columns.txt", "w")

word_array = []
pos_array = []
case = []
for i, line in enumerate(t):
   if i < 475000:
       line_split = line.split()
       len_split_1 = len(line_split[1])
       if line_split[1][len_split_1-1].isdigit() and line_split[1][len_split_1-2].isdigit():
           line_split[1] = line_split[1][0:len_split_1-2]
       line_split[1] = re.sub(r'[^a-zA-Z0-9]','', line_split[1])
       word_array.append(line_split[0].lower())
       pos_array.append(line_split[1])
       if line_split[0].isupper():
           case.append([1,0,0])
       if line_split[0].islower():
           case.append([0,1,0])
       else:
           case.append([0,0,1])
s = pd.Series(word_array)
s1 = pd.Series(pos_array)
print(case)
one_hot_word_array = pd.get_dummies(s)
one_hot_pos_array = pd.get_dummies(s1)
w.write("\n".join(one_hot_word_array.columns))
p.write("\n".join(one_hot_pos_array.columns))

t.close()
w.close()
p.close()
