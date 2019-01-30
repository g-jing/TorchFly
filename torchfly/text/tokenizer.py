import spacy
import numpy as np
from typing import List


class WordTokenizer():
    "Word Tokenizer"

    def __init__(self):
        self.tokenizer = spacy.load("en")

    def __call__(self, text):
        return self.tokenize(text)

    def tokenize(self, text: str)->List:
        tokens = self.tokenizer(text)
        return [str(t) for t in tokens]


class CharTokenizer():
    "Character Tokenizer based on UTF-8"

    @staticmethod
    def tokenize(tokens: List[str])->List[np.ndarray]:
        "Input must be list of string"
        final_res = []
        for t in tokens:
            final_res.append(np.array([c for c in t.encode("utf-8")]))
        return final_res
