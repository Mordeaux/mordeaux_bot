
"""
This module contains a class for making a generator out of the data
that has been seen by a TweetModel.
"""

from random import random

class TweetModelGenerator(object):
    """
    This class is a generator used to generate output according
    to the observed probabilities of a TweetModel.
    """
    def __init__(self, tweet_model):
        # `self.state` is a list of tokens that
        #  have been output thusfar.
        self.state = ['*', '*', '*', '*']
        self.tweet_model = tweet_model

    def _get_likelihoods(self, n, markov_dict):
        """
        Helper function to get a dict of likelihoods
        from a MarkovDict object.
        """
        model = self.tweet_model
        # Get the current state of the generator
        state = ' '.join(self.state[-n:])
        return markov_dict.get_max_likelihoods(
                state,
                discount=model.DISCOUNT)

    @staticmethod
    def _weighted_random_choice(tokens):
        """
        Takes a dict where keys are tokens, and values
        are the relative probabilities, and chooses one
        proportionally.
        """
        rand = random() * sum(tokens[token] for token in tokens)
        for token, weight in tokens.iteritems():
            rand -= weight
            if rand < 0:
                return token

    @staticmethod
    def _backoff(markov_dicts):
        """
        Takes a list of MarkovDicts and gives you a token,
        using a weighted random choice. If the first
        MarkovDict randomly selects the missing probability
        mass, it will back off to the next one.
        """
        # Just to shorten lines of code
        tmg = TweetModelGenerator
        markov_dict = markov_dicts[0]
        token = tmg._weighted_random_choice(markov_dict)
        if token == '$MPM':
            return tmg._backoff(markov_dicts[1:])
        else:
            return token

    def __iter__(self):
        model = self.tweet_model
        while True:
            bigrams = self._get_likelihoods(1, model.bigrams)
            trigrams = self._get_likelihoods(2, model.trigrams)
            quadgrams = self._get_likelihoods(3, model.quadgrams)
            token = self._backoff([
                quadgrams,
                trigrams,
                bigrams,
                model.unigrams])
            if token == self.tweet_model.EOS:
                raise StopIteration
            else:
                self.state.append(token)
                yield token


