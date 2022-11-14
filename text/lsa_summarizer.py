import math
from warnings import warn
from operator import attrgetter

import numpy as np
import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize
from numpy.linalg import svd 
from nltk.corpus import stopwords

from collections import namedtuple


SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))

class LsaSummarizer:
    MIN_DIMENSIONS = 3
    REDUCTION_RATIO = 1/1

    def __init__(self) -> None:
        self.stop_words = list(stopwords.words('english'))


    def __call__(self, document, sentences_count):

        dictionary = self.create_dictionary(document)
        
        if not dictionary:
            return ()

        sentences = sent_tokenize(document)

        matrix = self.create_matrix(document, dictionary)
        matrix = self.compute_term_frequency(matrix)
        u, sigma, v = svd(matrix, full_matrices=False)

        ranks = self.compute_ranks(sigma, v)
        return self.get_best_sentences(sentences, sentences_count, ranks)


    def create_dictionary(self, document):
        words = word_tokenize(document)
        words = map(self.normalize_word, words)
        unique_words = set(w for w in words if w not in self.stop_words)
        return dict((w, i) for i, w in enumerate(unique_words))


    def create_matrix(self, document, dictionary):
        """
        Creates matrix of shape where cells
        contains number of occurences of words (rows) in senteces (cols).
        """
        sentences = sent_tokenize(document)
        words_count = len(dictionary)
        sentences_count = len(sentences)
        if words_count < sentences_count:
            warn(f"Matrix might not be ful rank. # of words {words_count} is lower than # of sentences {sentences_count}. LSA algorithm may not work properly.")

        matrix = np.zeros((words_count, sentences_count))
        for col, sentence in enumerate(sentences):
            words = word_tokenize(sentence)
            for word in words:
                # only valid words is counted (not stop-words, ...)
                if word in dictionary:
                    row = dictionary[word]
                    matrix[row, col] += 1

        return matrix


    def compute_term_frequency(self, matrix, smooth=0.4):
        assert 0.0 <= smooth < 1.0

        max_word_frequencies = np.max(matrix, axis=0, keepdims=True)
        max_word_frequencies[max_word_frequencies==0] = 1

        frequency = matrix / max_word_frequencies
        matrix = smooth + (1.0 - smooth) * frequency
        
        return matrix


    def compute_ranks(self, sigma, v_matrix):
        assert len(sigma) == v_matrix.shape[0]

        dimensions = max(LsaSummarizer.MIN_DIMENSIONS,
            int(len(sigma)*LsaSummarizer.REDUCTION_RATIO))
        powered_sigma = tuple(s**2 if i < dimensions else 0.0
            for i, s in enumerate(sigma))

        ranks = []
        
        for column_vector in v_matrix.T:
            rank = sum(s*v**2 for s, v in zip(powered_sigma, column_vector))
            ranks.append(math.sqrt(rank))

        return ranks


    def normalize_word(self, word):
        return word.lower()


    def get_best_sentences(self, sentences, count, rating):

        infos = (SentenceInfo(s, o, rating[o]) for o, s in enumerate(sentences))

        # sort sentences by rating in descending order
        infos = sorted(infos, key=attrgetter("rating"), reverse=True)
        # get `count` first best rated sentences
        infos = infos[:min(count, len(infos))]
        # sort sentences by their order in document
        infos = sorted(infos, key=attrgetter("order"))

        return tuple(i.sentence for i in infos)