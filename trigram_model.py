import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Spring 2026
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1
    """
    padded = ['START'] * (n - 1) + sequence + ['STOP']
    return [tuple(padded[i:i+n]) for i in range(len(padded) - n + 1)]


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts.
        """

        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        self.total_words = 0

        for sentence in corpus:
            for unigram in get_ngrams(sentence, 1):
                self.unigramcounts[unigram] += 1
                self.total_words += 1
            for bigram in get_ngrams(sentence, 2):
                self.bigramcounts[bigram] += 1
            for trigram in get_ngrams(sentence, 3):
                self.trigramcounts[trigram] += 1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        bigram_prefix = trigram[:2]
        if self.bigramcounts[bigram_prefix] == 0:
            return 1.0 / len(self.lexicon)
        return self.trigramcounts[trigram] / self.bigramcounts[bigram_prefix]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        unigram_prefix = (bigram[0],)
        if self.unigramcounts[unigram_prefix] == 0:
            return 1.0 / len(self.lexicon)
        return self.bigramcounts[bigram] / self.unigramcounts[unigram_prefix]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        return self.unigramcounts[unigram] / self.total_words

    def generate_sentence(self,t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        result = []
        prev1 = 'START'
        prev2 = 'START'
        for _ in range(t):
            candidates = []
            probs = []
            for trigram, count in self.trigramcounts.items():
                if trigram[0] == prev1 and trigram[1] == prev2:
                    candidates.append(trigram[2])
                    probs.append(self.raw_trigram_probability(trigram))
            if not candidates:
                break
            total = sum(probs)
            probs = [p / total for p in probs]
            word = random.choices(candidates, weights=probs, k=1)[0]
            result.append(word)
            if word == 'STOP':
                break
            prev1 = prev2
            prev2 = word
        return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return (lambda1 * self.raw_trigram_probability(trigram) +
                lambda2 * self.raw_bigram_probability(trigram[1:]) +
                lambda3 * self.raw_unigram_probability((trigram[2],)))
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        log_prob = 0.0
        for trigram in trigrams:
            prob = self.smoothed_trigram_probability(trigram)
            log_prob += math.log2(prob)
        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """
        total_log_prob = 0.0
        total_words = 0
        for sentence in corpus:
            total_log_prob += self.sentence_logprob(sentence)
            total_words += len(sentence) + 1
        l = total_log_prob / total_words
        return 2 ** (-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0

        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total += 1
            if pp1 <= pp2:
                correct += 1

        for f in os.listdir(testdir2):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            total += 1
            if pp2 <= pp1:
                correct += 1

        return correct / total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("Perplexity:", pp)

    acc = essay_scoring_experiment("hw1_data/hw1_data/ets_toefl_data/train_high.txt",
                                  "hw1_data/hw1_data/ets_toefl_data/train_low.txt",
                                  "hw1_data/hw1_data/ets_toefl_data/test_high",
                                  "hw1_data/hw1_data/ets_toefl_data/test_low")
    print("Essay scoring accuracy:", acc)

