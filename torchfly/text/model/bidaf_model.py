import torch
import torch.nn as nn

from .models import CharCNN

class TextEmbedder(nn.Module):
    def __init__(self):
        super(TextEmbedder, self).__init__()
        self.word_embed = nn.Embedding(97914, 100, padding_idx=0)
        self.char_embed = CharCNN()
        
    def forward(self, word, char):
        w = self.word_embed(word)
        c = self.char_embed(char)
        x = torch.cat([c, w], dim=2)
        return x