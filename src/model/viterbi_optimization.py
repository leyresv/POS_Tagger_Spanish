import numpy as np
import math


def viterbi_forward(pos_tags, tag_counts, transition_matrix, emission_matrix, vocab, words):

    num_tags = len(tag_counts)

    # Initialize matrices
    state_prob_matrix = np.zeros((num_tags, len(words)))
    bos_idx = pos_tags.index("<BOS>")
    state_prob_matrix[bos_idx, 0] = 1

    backtrack_matrix = np.zeros((num_tags, len(words)), dtype=int)
    eos_idx = pos_tags.index("<EOS>")
    backtrack_matrix[bos_idx, 0] = eos_idx

    # For each word in the sequence (word 0 already initialized)
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
                prob = state_prob_matrix[prev_tag_idx, word_idx - 1] + math.log(
                    transition_matrix[prev_tag_idx, tag_idx]) + math.log(
                    emission_matrix[tag_idx, vocab[words[word_idx]]])

                # If that probability is greater than the current best probability,
                if prob > best_prob:
                    # assign this probability as best probability
                    best_prob = prob
                    # assign the previous tag index as the best path
                    best_path = prev_tag_idx

            state_prob_matrix[tag_idx, word_idx] = best_prob
            backtrack_matrix[tag_idx, word_idx] = best_path

    return state_prob_matrix, backtrack_matrix


def viterbi_backward(state_prob_matrix, backtrack_matrix, pos_tags, words):
    num_words = backtrack_matrix.shape[1]
    num_tags = len(pos_tags)
    pred_idx = [None] * num_words
    pred = [None] * num_words
    best_prob_last_word = float(" -inf")

    for tag_idx in range(num_tags):
        # Find the highest probability from that column
        if state_prob_matrix[tag_idx, -1] > best_prob_last_word:
            best_prob_last_word = state_prob_matrix[tag_idx, -1]
            pred_idx[num_words - 1] = tag_idx

    pred[num_words - 1] = pos_tags[tag_idx]

    # Iterate backwards through the words. For each word:
    for word_idx in range(num_words - 1, -1, -1):
        # Get the tag index with the highest probability in that column
        tag_idx = np.argmax(state_prob_matrix[:, word_idx])
        pos_tag = backtrack_matrix[tag_idx, word_idx]

        # Get the previous word's tag index
        pred_idx[word_idx - 1] = backtrack_matrix[pos_tag, word_idx]

        # Get the previous word's tag
        pred[word_idx - 1] = pos_tags[pos_tag]

    return pred
