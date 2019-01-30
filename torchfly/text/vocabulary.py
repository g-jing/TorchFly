import collections


class Vocabulary():
    """
    Vocabulary should store correspondences for all words,
    including PAD, UNK, BOS, EOS.
    """

    def __init__(self, itow):
        "This should be exact match for index to word"
        self.itow = itow

    @classmethod
    def from_file(self, file_name):
        raise NotImplementedError

    # def __getstate__(self):
    #     "unfinished"
    #     raise NotImplementedError
    #     # return {'itos': self.itos}

    # def __setstate__(self, state: dict):
    #     "unfinished"
    #     self.itow = state['itow']
    #     self.stoi = collections.defaultdict(
    #         int, {v: k for k, v in enumerate(self.itow)})
    #     raise NotImplementedError

    # Decided to remove the following lines
    #
    # def numericalize(self, t):
    #     "Convert a list of tokens `t` to their ids."
    #     return [self.stoi[w] for w in t]

    # def numericalize_all(self, tokens):
    #     "Convert a list of sentences of tokens to their ids."
    #     return [self.numericalize(t) for t in tokens]

    # def textify(self, nums, sep=' '):
    #     "Convert a list of `nums` to their tokens."
    #     return sep.join([self.itos[i] for i in nums])
