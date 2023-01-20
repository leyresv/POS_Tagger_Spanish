import pickle
import os
import json

from src.model.hidden_markov_model import HMM

cur_file_path = os.path.dirname(os.path.realpath(__file__))


def load_train_data():
    train_data = json.load(open(os.path.join(cur_file_path, "../../data/train_word_tag.json")))
    vocab_index = json.load(open(os.path.join(cur_file_path, "../../data/vocab_index.json")))
    pos_tags = json.load(open(os.path.join(cur_file_path, "../../data/pos_tags.json")))
    with open(os.path.join(cur_file_path, "../../data/vocab_counts.json"), "r") as f:
        for i in f.readlines():
            dic = i  # string
    vocab_counts = eval(dic)
    return train_data, vocab_counts, vocab_index, pos_tags


if __name__ == "__main__":
    # Load train data
    train_data, vocab_counts, vocab_index, pos_tags = load_train_data()
    # Train HMM
    hmm = HMM()
    hmm.train(train_data, vocab_counts, vocab_index, pos_tags)
    # Save HMM
    pickle.dump(hmm, open(os.path.join(cur_file_path, "../../models/hmm_model.pkl"), "wb"))
