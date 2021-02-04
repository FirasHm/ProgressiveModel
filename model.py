import torch.nn as nn
import torch.nn.functional as F
from modules import Lstm, Attention

class AttRNN(nn.Module):
    def __init__(self, params, vocab):
        super(AttRNN, self).__init__()
        self.hidden_size = params.hidden_size

        self.enc = Lstm(params, vocab)
        self.dec = Lstm(params, vocab)
        self.attn = Attention(attention_size=self.hidden_size)

    def forward(self, x):
        # encoder input
        output, (h_n, _) = self.enc(x)
        
        # decoder input
        attn_output = self.attn(h_n, self.hidden_size)
        output = self.dec(attn_output)

        return output


class ProgModel(nn.Module):
    def __init__(self, params, dict):
        super(ProgModel, self).__init__()
        self.lstm = nn.LSTM(input_size=params.input_size, hidden_size=params.hidden_size, num_layers=1, bidirectional=True, )
        self.type = params.ide_type
        self.context_gate = None ###
        if self.type != 'c' and self.type != 't':
            print("Wrong type")
            return 

        

    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        """
        returns loss 
        """

        if self.type == 't':
            pass
        else:
            pass
        return

    def expand(self, pre_model: ProgModel, ):
        """
        lateral connection between current model and previous model
        """
        h_bar = h + F.sigmoid(torch.matmul(V_t, h))
        



    

class RZT(nn.Module):
    def __init___(self, ):
        super(model, self).__init__()
        self.encoder = nn.GRU(input_size=, hidden_size=, num_layers=2, bidirectional=True, )
        self.attention