import numpy as np
import math


def viterbi_initialize(pos_tags, tag_counts, transition_matrix, emission_matrix, vocab, words):

    num_tags = len(tag_counts)

    best_tagseq_probabilities = np.zeros((num_tags, len(words)))
    best_paths = np.zeros((num_tags, len(words)), dtype=int)

    bos_idx = pos_tags.index("<BOS>")

    for tag_idx in range(num_tags):
        if transition_matrix[(bos_idx, tag_idx)] == 0:
            best_tagseq_probabilities[tag_idx, 0] = float("-inf")

        else:
            best_tagseq_probabilities[tag_idx, 0] = math.log(transition_matrix[bos_idx, tag_idx]) + math.log(
                emission_matrix[tag_idx, vocab[words[0]]])

    return best_tagseq_probabilities, best_paths


def viterbi_forward(transition_matrix, emission_matrix, best_tagseq_probabilities, best_paths, vocab, words):

    num_tags = best_tagseq_probabilities.shape[0]

    # For each word in the corpus (word 0 already initialized)
    for word_idx in range(1, len(words)):

        # For each POS tag type that this word could be
        for tag_idx in range(num_tags):

            best_prob = float("-inf")
            best_path = None

            # For each POS tag that the previous word could be:
            for prev_tag_idx in range(num_tags):

                # compute the probability that the previous word had a given POS tag,
                # that the current word has a given POS tag,
                # and that the POS tag would emit this current word
                prob = best_tagseq_probabilities[prev_tag_idx, word_idx - 1] + math.log(
                    transition_matrix[prev_tag_idx, tag_idx]) + math.log(
                    emission_matrix[tag_idx, vocab[words[word_idx]]])

                if prob > best_prob:
                    best_prob = prob
                    best_path = prev_tag_idx

            best_tagseq_probabilities[tag_idx, word_idx] = best_prob
            best_paths[tag_idx, word_idx] = best_path


def viterbi_backward(best_tagseq_probabilities, best_paths, pos_tags, words):
    num_words = best_paths.shape[1]
    num_tags = len(pos_tags)
    pred_idx = [None] * num_words
    pred = [None] * num_words
    best_prob_last_word = float(" -inf")

    for tag_idx in range(num_tags):
        # Find the highest probability from that column
        if best_tagseq_probabilities[tag_idx, -1] > best_prob_last_word:
            best_prob_last_word = best_tagseq_probabilities[tag_idx, -1]
            pred_idx[num_words - 1] = tag_idx

    pred[num_words - 1] = pos_tags[tag_idx]

    # Iterate backwards through the words. For each word:
    for word_idx in range(num_words - 1, -1, -1):
        # Get the tag index with the highest probability in that column
        tag_idx = np.argmax(best_tagseq_probabilities[:, word_idx])
        pos_tag = best_paths[tag_idx, word_idx]

        # Get the previous word's tag index
        pred_idx[word_idx - 1] = best_paths[pos_tag, word_idx]

        # Get the previous word's tag
        pred[word_idx - 1] = pos_tags[pos_tag]

    return pred
