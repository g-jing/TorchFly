import torch
import torch.nn as nn

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self._module = module
        
    def forward(self, x):
        """Shape -> (batch_size, time_steps, *shapes)"""
        bs = x.shape[0]
        ts = x.shape[1]
        x = x.view(bs*ts, *x.shape[2:])
        x = self._module(x)
        x = x.view(bs, ts, *x.shape[1:])
        return x


class CharCNN(nn.Module):
    def __init__(self):
        super(CharCNN, self).__init__()
        self.char_embed = nn.Embedding(262, 16, padding_idx=0)
        self.conv1 = TimeDistributed(nn.Conv1d(16, 100, kernel_size=(5,), stride=(1,)))
        self.activ = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        "Takes Packed Sequence as Input"
        # (pack_len, seq_len)
        x = self.char_embed(x)
        # (pack_len, seq_length, in_channels) 
        x = x.transpose(2,3)
        x = self.activ(self.conv1(x))
        x = x.max(-1)[0]
        # (pack_len, out_channels) 
        return x

class Highway(nn.Module):
    "From AllenNLP"
    
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 1):
        super(Highway, self).__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim * 2)
                                            for _ in range(num_layers)])
        self._activ = nn.ReLU()
        
        # make bias positive to carry forward
        for layer in self._layers:
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, x):
        for layer in self._layers:
            f_x, gate = layer(x).chunk(2, dim=-1)
            f_x = self._activ(f_x)
            gate = torch.sigmoid(gate)
            x = gate * x + (1 - gate) * f_x
        return x