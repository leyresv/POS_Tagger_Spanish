import json
import os
from datasets import load_dataset
from collections import Counter

cur_file_path = os.path.dirname(os.path.realpath(__file__))


def create_vocab(dataset):
    #dataset = load_dataset("PlanTL-GOB-ES/UD_Spanish-AnCora", split="train")
    pos_tags = dataset.features["upos_tags"].feature.names
    print(pos_tags)
    pos_tags.append("<BOS>")
    pos_tags.append("<EOS>")
    print(pos_tags)

    # Get the frequency of every word-tag pair in the dataset
    word_tag = []
    for sentence in dataset:
        word_tag.append(("<BOS>", "<BOS>"))
        for word, tag in (zip(sentence["tokens"], sentence["upos_tags"])):
            word_tag.append((word, tag))
      #  word_tag.append(("<BOS>", pos_tags.index("<BOS>")))
        word_tag.append(("<EOS>", "<EOS>"))
    vocab_counts_all = Counter(sorted(word_tag))

    # Replace low-frequency terms by unknown token
    vocab_counts = {("<UNK>", pos_tags.index("X")): 0}
    for tup, count in vocab_counts_all.items():
        if count < 2:
            vocab_counts[("<UNK>", pos_tags.index("X"))] += count
        else:
            vocab_counts[tup] = count

    # Encode words of vocabulary
    vocab_words = sorted(set([word for word, tag in vocab_counts.keys()]))
    vocab_index = {word: i for i, word in enumerate(vocab_words)}

    return vocab_counts, vocab_index, pos_tags


def preprocess_dataset(dataset, vocab, pos_tags):
    """
    Preprocess UD_Spanish-AnCora dataset

    :param dataset: spanish POS tags dataset
    :param vocab: words in the training dataset
    :param pos_tags: list of possible POS tags
    :return: list of (word, POS tag)
    """
    word_tag = []
    for sentence in dataset:
        word_tag.append(("<BOS>", "<BOS>"))
        for word, tag in (zip(sentence["tokens"], sentence["upos_tags"])):
            if word not in vocab:
                word = "<UNK>"
            word_tag.append((word, pos_tags[tag]))
        word_tag.append(("<EOS>", "<EOS>"))

    return word_tag


if __name__ == "__main__":

    # Import datasets
    train_dataset = load_dataset("PlanTL-GOB-ES/UD_Spanish-AnCora", split="train")
    test_dataset = load_dataset("PlanTL-GOB-ES/UD_Spanish-AnCora", split="test")

    # Create vocabulary
    vocab_counts, vocab_index, pos_tags = create_vocab(train_dataset)

    # Write vocabulary to files
    with open(os.path.join(cur_file_path, "../../data/vocab_counts.json"), 'w') as f:
        f.write(str(vocab_counts))
    json.dump(vocab_index, open(os.path.join(cur_file_path, "../../data/vocab_index.json"), 'w'))
    json.dump(pos_tags, open(os.path.join(cur_file_path, "../../data/pos_tags.json"), 'w'))

    # Process train and test data
    train_word_tag = preprocess_dataset(train_dataset, vocab_index, pos_tags)
    test_word_tag = preprocess_dataset(test_dataset, vocab_index, pos_tags)

    # Write train and test data to files
    json.dump(train_word_tag, open(os.path.join(cur_file_path, "../../data/train_word_tag.json"), 'w'))
    json.dump(test_word_tag, open(os.path.join(cur_file_path, "../../data/test_word_tag.json"), 'w'))

