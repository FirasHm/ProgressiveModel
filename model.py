'''
실제 학습하고자 하는 모델을 기술
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import EncoderRNN, AttnDecoderRNN

class ProgModel(nn.Module):
    def __init__(self, params, dict):
        super(ProgModel, self).__init__()
        self.enc = EncoderRNN()
        self.dec = AttnDecoderRNN()
        self.type = params.ide_type
        self.context_gate = None ###
        if self.type != 'c' and self.type != 't':
            print("Wrong type")
            return 


    def forward(self, *input: Any, **kwargs: Any):
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