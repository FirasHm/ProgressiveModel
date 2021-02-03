import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# A Progressive Model to Enable Continual Learning for Semantic Slot Filling

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