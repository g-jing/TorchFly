import numpy as np
import collections

from .vocabulary import Vocabulary
from typing import Any, Dict, List


class WordIndexer():
    "Word 2 Index"
    def __init__(self, vocab: Vocabulary, rules: Dict[str, int]):
        """
        rules: map the speicic token into a word in the vocabulary.
        """
        self.vocab = vocab
        self.rules = rules

        self.wtoi = collections.defaultdict(lambda: self.rules["UNK"],
                                    {v: k for k, v in enumerate(self.vocab.itow)})
                            
        # add rules into wtoi
        for rule in rules.keys():
            self.wtoi[rule] = rules[rule]

    def numericalize(self, tokens: List[str])->List[np.ndarray]:
        "Return the numercalized results"
        return [self.wtoi[w] for w in tokens]

    def textify(self):
        raise NotImplementedError


class CharIndexer():
    "Not implemented yet"
    def __init__(self):
        pass

    def numericalize(self, tokens: List[np.ndarray])->List[np.ndarray]:
        raise NotImplementedError

    def textify(self):
        raise NotImplementedError
