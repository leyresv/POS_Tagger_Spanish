import pickle
import os
import json
import numpy as np
from collections import defaultdict
from datasets import load_dataset

from src.data.preprocess_data import create_vocab, preprocess_dataset
from src.model.viterbi_optimization import viterbi_initialize, viterbi_forward, viterbi_backward

cur_file_path = os.path.dirname(os.path.realpath(__file__))

class HMM:

    def __init__(self):
        self.vocab_counts = None
        self.vocab_index = None
        self.pos_tags = None

        # Dictionary of (prev_tag, tag) : counts --> counts the amount of times that each tag pair appears
        self.transition_counts = defaultdict(int)
        # Dictionary of (tag, word) : counts --> counts the amount of times that each tag-word appears
        self.emission_counts = defaultdict(int)
        # Dictionary of (tag): counts --> counts the amount of times that a certain tag appears
        self.tag_counts = defaultdict(int)

        # Matrix with probability of a POS tag given another POS tag (from previous word)
        self.transition_matrix = None
        # Matrix with probability of a word given its POS tag
        self.emission_matrix = None

        self.best_tagseq_probabilities = None
        self.best_paths = None

    def populate_state_dictionaries(self, word_tag):
        """
        Populate transitions, emission and state count dictionaries
        :param word_tag: list of (word, pos_tag) tuples
        """
        # Initialize previous tag with the beginning of sentence state
        prev_tag = "<EOS>"
        self.tag_counts[prev_tag] += 1

        # For each word, tag pair
        for word, tag in word_tag:
            # Increase transition, emission and tag counts
            self.transition_counts[(prev_tag, tag)] += 1
            self.emission_counts[(tag, word)] += 1
            self.tag_counts[tag] += 1

            # Update prev_tag with current tag for next iteration
            prev_tag = tag

    def generate_transition_matrix(self, alpha=0.001):
        """
        Compute matrix with probability of a POS tag (hidden state) given another POS tag (previous hidden state).

        :param alpha: smoothing parameter
        """

        print(self.tag_counts.keys())
        tags_list = sorted(self.tag_counts.keys())
        print(tags_list)
        num_tags = len(tags_list)

        # Initialize transition_matrix
        self.transition_matrix = np.zeros((num_tags, num_tags))

        # For each row of the matrix
        for prev_tag_idx in range(num_tags):

            # For each column of the row
            for tag_idx in range(num_tags):

                count = 0
                key = (tags_list[prev_tag_idx], tags_list[tag_idx])
                # If transition prev_tag -> tag exists in training data, get its total count
                if key in self.transition_counts:
                    count = self.transition_counts[key]

                # Get amount of times that the previous tag appears
                count_prev_tag = self.tag_counts[tags_list[prev_tag_idx]]

                # Update transition matrix with P(tag | prev_tag)
                self.transition_matrix[prev_tag_idx, tag_idx] = (count + alpha) / (count_prev_tag + alpha * num_tags)

    def generate_emission_matrix(self, alpha=0.001):
        """
        Compute matrix with probability of a word (observed event) given its POS tag (hidden state)

        :param alpha: smoothing parameter
        """

        tags_list = sorted(self.tag_counts.keys())
        num_tags = len(tags_list)

        words_list = list(self.vocab_index)
        num_words = len(words_list)

        # Initialize emission matrix
        self.emission_matrix = np.zeros((num_tags, num_words))

        # For each row of the matrix
        for tag_idx in range(num_tags):

            # For each column of the row
            for word_idx in range(num_words):

                count = 0
                key = (tags_list[tag_idx], words_list[word_idx])
                # If emission tag -> word exists in training data, get its total count
                if key in self.emission_counts:
                    count = self.emission_counts[key]

                # Get amount of times that the tag appears
                count_tag = self.tag_counts[tags_list[tag_idx]]

                # Update emission matrix with P(word | tag)
                self.emission_matrix[tag_idx, word_idx] = (count + alpha) / (count_tag + alpha * num_words)

    def train(self, word_tag, vocab_counts, vocab_index, pos_tags):
        self.vocab_counts, self.vocab_index, self.pos_tags = vocab_counts, vocab_index, sorted(pos_tags)
        self.populate_state_dictionaries(word_tag)
        self.generate_transition_matrix()
        self.generate_emission_matrix()

    def predict(self, words):
        """
        Predict pos tags with Viterbi optimization

        :param words: list of words
        :return: list of predicted tags
        """
        # Initialize Viterbi algorithm
        self.best_tagseq_probabilities, self.best_paths = viterbi_initialize(self.pos_tags,
                                                                             self.tag_counts,
                                                                             self.transition_matrix,
                                                                             self.emission_matrix,
                                                                             self.vocab_index,
                                                                             words)

        # Forward pass
        viterbi_forward(self.transition_matrix,
                        self.emission_matrix,
                        self.best_tagseq_probabilities,
                        self.best_paths,
                        self.vocab_index,
                        words)

        # Backward pass
        predicted_tags = viterbi_backward(self.best_tagseq_probabilities, self.best_paths, sorted(self.pos_tags), words)

        return predicted_tags

    def naive_predict(self, words):
        """
        Naïve POS tag prediction (without Viterbi optimization).
        To each word, assign POS tag with the highest emission count.

        :param words: list of words
        :return: list of predicted tags
        """
        predicted_tags = []
        for word in words:
            # print(word)
            best_tag = ""
            highest_count = 0
            if word in self.vocab_index:
                for tag in self.pos_tags:
                    count = self.emission_counts[(tag, word)]
                    if count > highest_count:
                        highest_count = count
                        best_tag = tag
            predicted_tags.append(best_tag)
        return predicted_tags

    def get_vocab(self):
        return self.vocab_index

