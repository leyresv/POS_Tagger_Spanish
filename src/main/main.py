import os
import pickle

cur_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(cur_file_path, "../data")


if __name__ == "__main__":
    pos_tagger = pickle.load(open(os.path.join(cur_file_path, "../../models/hmm_model.pkl"), "rb"))
    print("Welcome to this POS tagger for Spanish.")
    sentence = input("Please, insert your sentence in Spanish: ")
    vocab_index = pos_tagger.get_vocab()
    words = ["<BOS>"]
    for word in sentence.split():
        if word in vocab_index:
            words.append(word)
        else:
            words.append("<UNK>")
    words.append("<EOS>")
    print(words)

    tags = pos_tagger.predict(words)

    print("{: >20} {: >20}".format("Word",  "Tag"))
    for word, tag in zip(words[1:-1], tags[1:-1]):
        print("{: >20} {: >20}".format(word, tag))
       # print(word, "\t", tag)
