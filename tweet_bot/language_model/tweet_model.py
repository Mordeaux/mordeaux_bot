"""
This module contains a class for creating a tweet
language model.
"""

import os
from collections import defaultdict, Counter
from copy import deepcopy
from random import random

from nltk.tokenize import TweetTokenizer

from tweet_model_generator import TweetModelGenerator

# This constant determines the discount used by the language model.
DISCOUNT = float(os.environ.get('DISCOUNT', 0.6))


class MarkovDict(defaultdict):
    """
    This is a subclass of collections.defaultdict which defaults all
    keys to collections.Counter, and provides some helper methods for
    the purposes of modelling a Markov chain.

    >>> tweet_model = TweetModel()
    >>> tweet_model.train('This is a tweet')
    >>> tweet_model.train('This is not')
    >>> tweet_model.train('This could be a tweet')
    >>> tweet_model.quadgrams
    defaultdict(<class 'collections.Counter'>, {u'* This could': Counter({u'be': 1}), u'* This is': Counter({u'a': 1, u'not': 1}), u'could be a': Counter({u'tweet': 1}), u'* * This': Counter({u'is': 2, u'could': 1}), u'This is a': Counter({u'tweet': 1}), u'be a tweet': Counter({'$eos': 1}), u'is a tweet': Counter({'$eos': 1}), u'This is not': Counter({'$eos': 1}), u'This could be': Counter({u'a': 1}), '* * *': Counter({u'This': 3})})
    """

    def __init__(self, *args):
        super(MarkovDict, self).__init__(Counter)

    def _get_total(self, n_gram):
        """
        Returns an integer representing the total count of all times a
        particular n_gram has been seen.

        >>> tweet_model = TweetModel()
        >>> tweet_model.train('This is a very good tweet')
        >>> tweet_model.train('This is not a very very good tweet')
        >>> tweet_model.bigrams._get_total('This')
        2
        >>> tweet_model.bigrams._get_total('very')
        3
        >>> tweet_model.bigrams._get_total('twitter')
        0
        >>> tweet_model.bigrams
        defaultdict(<class 'collections.Counter'>, {u'a': Counter({u'very': 2}), u'good': Counter({u'tweet': 2}), u'This': Counter({u'is': 2}), u'is': Counter({u'a': 1, u'not': 1}), '*': Counter({u'This': 2}), u'very': Counter({u'good': 2, u'very': 1}), u'not': Counter({u'a': 1}), u'tweet': Counter({'$eos': 2})})
        """
        total = sum(self[n_gram].values())
        if not total:
            del self[n_gram]
        return total

    def get_max_likelihoods(self, n_gram, limit=None, discount=0):
        """
        Takes the last seen n_gram and returns a dict with tokens seen
        following that n_gram as keys, and their probabilities as
        values. If a limit of `x` is provided only the top `x` most
        likely tokens will be returned.

        >>> tweet_model = TweetModel()
        >>> tweet_model.train('This is a very good tweet')
        >>> tweet_model.train('This is not a very good tweet')
        >>> tweet_model.bigrams.get_max_likelihoods('is')
        {u'a': 0.5, u'not': 0.5}
        """
        # Make sure that the discount is a float so that
        # other numbers will coerce to floats.
        discount = float(discount)
        total_count = self._get_total(n_gram)
        tokens = self[n_gram].most_common(limit)
        try:
            probabilities = dict([(token, (count - discount) / total_count)
                                     for token, count in tokens])
            if discount:
                # Need to return the missing probability from smoothing.
                probabilities['$MPM'] = discount * len(self[n_gram]) / total_count
            return probabilities
        # If the n_gram has not been seen before:
        except ZeroDivisionError:
            return {'$MPM': 1}

    @staticmethod
    def _merge(base, markov_dict):
        """
        This method ensures that two MarkovDicts can be properly
        combined to return a new MarkovDict, allowing the user
        to dynamically continue adding new information to the model.
        """
        # Make deepcopy to preserve original MarkovDict
        ret = deepcopy(base)
        # Iterate over the keys and update the counters
        for token in markov_dict:
            ret[token].update(markov_dict[token])
        return ret

    def __add__(self, other):
        """
        This overloads the addition operator so the user can add two
        MarkovDicts together and get one MarkovDict containing the
        resulting total values.
        """
        if type(other) == type(self):
            return self._merge(self, other)
        else:
            error = 'Objects of type: {} can only be'
            ' added to others of the same type'.format(type(self))
            raise NotImplementedError(error)


class TweetModel(object):
    """
    A language Model class for tweets.
    """
    # This token will mark the end of a string of input,
    # changing it will break the doctests.
    EOS = '$eos'
    TOKENIZER = TweetTokenizer()
    DISCOUNT = DISCOUNT

    def __init__(self):
        self.unigrams = Counter()
        self.bigrams = MarkovDict()
        self.trigrams = MarkovDict()
        self.quadgrams = MarkovDict()

    def train(self, text):
        """
        Takes a text as a string, and trains this TweetModel.
        """
        tokens = self.TOKENIZER.tokenize(text)

        bigrams = self._get_bigrams(tokens)
        trigrams = self._get_trigrams(tokens)
        quadgrams = self._get_quadgrams(tokens)

        self.unigrams.update(tokens)
        self.bigrams += bigrams
        self.trigrams += trigrams
        self.quadgrams += quadgrams

    def get_generator(self):
        return (token for token in TweetModelGenerator(self))

    @staticmethod
    def _get_n_grams(tokens, n):
        """
        Takes a list of tokens and returns a Counter object.
        >>> m = TweetModel()
        >>> tokens = ['one', 'two', 'three']
        >>> m._get_n_grams(tokens, 3)
        defaultdict(<class 'collections.Counter'>, {'one two three': Counter({'$eos': 1}), '* * one': Counter({'two': 1}), '* one two': Counter({'three': 1}), '* * *': Counter({'one': 1})})
        """
        tokens = (['*'] * n) + tokens + [TweetModel.EOS]
        n_grams = MarkovDict()
        for i in xrange(len(tokens) - n):
            prior_token = ' '.join(tokens[i:i+n])
            next_token = tokens[i+n]
            n_grams[prior_token].update([next_token])
        return n_grams

    @staticmethod
    def _get_bigrams(tokens):
        return TweetModel._get_n_grams(tokens, 1)

    @staticmethod
    def _get_trigrams(tokens):
        return TweetModel._get_n_grams(tokens, 2)

    @staticmethod
    def _get_quadgrams(tokens):
        return TweetModel._get_n_grams(tokens, 3)


if __name__ == '__main__':
    tweets = [
            """We all got a chicken duck woman thing waiting for us: https://www.youtube.com/watch?v=RySHDU""",
            """I pledge not to vote for hillary. Have you taken the pledge?""",
            """Thats the last straw. If drumpf wins in november im moving to stankonia.""",
            """Haha that's me on the left looking into the distance!""",
            """If youre ever feeling down about your life, read the "personal life" section of andy dick's wikipedia page.""",
            """Little kid telling his mom on the subway that if it gets windy enough he can fly. Shes polite but not buying it. Neither was i.""",
            """I need your help twitter, can anyone put me in touch with an accordion teacher?""",
            ]
    model = TweetModel()
    for tweet in tweets:
        model.train(tweet)

    for token in model.get_generator():
        print token,

