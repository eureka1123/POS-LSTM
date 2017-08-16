"""
main_dictionary["word2index"] = word2index
main_dictionary["pos2index"] = pos2index
main_dictionary["index2word"] = index2word
main_dictionary["index2pos"] = index2pos
main_dictionary["word_pos_pairs"] = word_pos_pairs
main_dictionary["encoded_input"] = word_array_encoded
main_dictionary["encoded_label"] = pos_array_encoded
"""

with open("data/main_dictionary_size21.txt", "r") as file:
    main_dict = eval(file.read())

def get_index_from_word(word):
    return main_dict["word2index"].get(word)

def get_index_from_pos(pos):
    return main_dict["pos2index"].get(pos)

def get_word_from_index(index):
    return main_dict["index2word"].get(index)

def get_pos_from_index(index):
    return main_dict["index2pos"].get(index)


word_pos_pairs = main_dict["word_pos_pairs"]
word_array = [[word[0] for word in sentence] for sentence in word_pos_pairs]
pos_array = [[word[1] for word in sentence] for sentence in word_pos_pairs]


