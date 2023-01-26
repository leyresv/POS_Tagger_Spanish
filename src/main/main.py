import os
import pickle
from spacy.lang.es import Spanish

cur_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(cur_file_path, "../data")

class POS_Tagger:

    def __init__(self):
        self.nlp = Spanish()
        self.nlp.add_pipe("sentencizer")
        self.pos_tagger = pickle.load(open(os.path.join(cur_file_path, "../../models/hmm_model.pkl"), "rb"))

    def preprocess_input(self, text):
        doc = self.nlp(text)
        words = []
        for sentence in doc.sents:
            words.append("<BOS>")
            for token in sentence:
                word = token.text
                if word not in self.pos_tagger.get_vocab():
                    word = "<UNK>"
                words.append(word)
            words.append("<EOS>")
        return words

    def tag(self, words):
        tags = self.pos_tagger.predict(words)
        return tags


if __name__ == "__main__":
    print("Welcome to this POS tagger for Spanish.")
    pos_tagger = POS_Tagger()
    text = input("Please, insert your sentence in Spanish ('Q' to quit): ")
    while text != "Q":
        words = pos_tagger.preprocess_input(text)
        tags = pos_tagger.tag(words)

        print("{: >20} {: >20}".format("Word",  "Tag"))
        for word, tag in zip(words, tags):
            if word not in ["<BOS>", "<EOS>"]:
                print("{: >20} {: >20}".format(word, tag))

        print()
        text = input("Please, insert your sentence in Spanish ('Q' to quit): ")

