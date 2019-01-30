import collections
import numpy as np

#pylint: disable=E0402
from ..tokenizer import WordTokenizer, CharTokenizer
from ..indexer import WordIndexer, CharIndexer

from typing import Any, Dict, List


class BiDAFVocab():
    def __init__(self, itow):
        "This should be exact match for index to word"
        self.itow = itow

    @classmethod
    def from_file(cls, file_name):
        with open(file_name, 'r+') as f:
            itow = [line for line in f.read().split('\n')]

        itow = itow[:-1]
        itow.insert(0, "@@PAD@@")

        return cls(itow)


class BiDAFCharIndexer():
    START = np.array([259])
    END = np.array([260, 0, 0, 0, 0, 0])
    PAD = np.array([0])

    @staticmethod
    def numericalize(tokens: List[np.ndarray])->List[np.ndarray]:
        return [np.concatenate([BiDAFCharIndexer.START, t+1, BiDAFCharIndexer.END]) for t in tokens]


class BiDAF_Pipeline():
    RULES = {'\n': 2609,
             '\r\n': 2609,
             '\r': 2609,
             'UNK': 1
             }

    def __init__(self, vocab_file_name: str):
        self.vocab = BiDAFVocab.from_file(vocab_file_name)
        self.word_tokenizer = WordTokenizer()
        self.word_indexer = WordIndexer(self.vocab, BiDAF_Pipeline.RULES)

    def __call__(self, text: str)->[List, List]:
        return self.process_one(text)

    def process_one(self, text: str)->Dict[str, Any]:
        "Returns word tokens and character tokens"

        word_tokens = self.word_tokenizer(text)
        char_tokens = CharTokenizer.tokenize(word_tokens)

        word_tokens = [t.lower() for t in word_tokens]
        # numericalize
        word_tokens = np.array(self.word_indexer.numericalize(word_tokens))
        char_tokens = BiDAFCharIndexer.numericalize(char_tokens)

        return {"word_tokens": word_tokens,
                "char_tokens": char_tokens}
