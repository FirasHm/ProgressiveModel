'''
모델 정의 시 필요한 클래스 정의
'''

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from lib import load_embedding_from_npy, load_embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class attrnn(nn.Module):
    def __init__(self):
        super(attrnn, self).__init__()
        self.enc = nn.LSTM(bidirectional=True)
        self.dec = nn.LSTM()
        
    def forward(self, x):
        enc_out, (h_e, c_e) = self.enc(x)
        dec_out, (h_d, c_d) = self.dec(enc_out)
        attn = self.attention()

    def attention(self):
        attn = torch.zeros_like(dec.hidden)
        for i, dec_hidden in enumerate(dec.hidden):
            # dot-product attention
            score = torch.matmul(enc.hidden, dec_hidden)
            e = F.softmax(score)
            attn[i] = torch.sum(torch.matmul(enc.hidden, e), axis=1)

        return attn

        
# https://github.com/zliucr/coach/blob/master/src/modules.py
class Lstm(nn.Module):
    def __init__(self, params, vocab):
        super(Lstm, self).__init__()
        self.n_layer = params.n_layer
        self.emb_dim = params.emb_dim
        self.n_words = vocab.n_words
        self.hidden_dim = params.hidden_dim
        self.dropout = params.dropout
        self.bidirection = params.bidirection
        self.freeze_emb = params.freeze_emb
        self.emb_file = params.emb_file

        # embedding layer
        self.embedding = nn.Embedding(self.n_words, self.emb_dim, padding_idx=PAD_INDEX)
        # load embedding
        if self.emb_file.endswith("npy"):
            embedding = load_embedding_from_npy(self.emb_file)
        else:
            embedding = load_embedding(vocab, self.emb_dim, self.emb_file)
        self.embedding.weight.data.copy_(torch.FloatTensor(embedding))
        
        # LSTM layers
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=self.n_layer, 
                        dropout=self.dropout, bidirectional=self.bidirection, batch_first=True)
        
       
    def forward(self, X):
        """
        Input:
            x: text (bsz, seq_len)
        Output:
            last_layer: last layer of lstm (bsz, seq_len, hidden_dim)
        """
        embeddings = self.embedding(X)
        embeddings = embeddings.detach() if self.freeze_emb else embeddings
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)

        # LSTM
        # lstm_output (batch_first): (bsz, seq_len, hidden_dim)
        lstm_output, (_, _) = self.lstm(embeddings)

        return lstm_output


# https://github.com/huggingface/torchMoji/blob/master/torchmoji/attlayer.py
class Attention(nn.Module):
    """
    Computes a weighted average of channels across timesteps (1 parameter pr. channel).
    """
    def __init__(self, attention_size, return_attention=False):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
            return_attention: If true, output will include the weight for each input token
                              used for the prediction
        """
        super(Attention, self).__init__()
        self.return_attention = return_attention
        self.attention_size = attention_size
        self.attention_vector = nn.Parameter(torch.FloatTensor(attention_size))
        torch.nn.init.uniform_(self.attention_vector.data, -0.01, 0.01)

    def __repr__(self):
        s = '{name}({attention_size}, return attention={return_attention})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, input_lengths, y_bin=None):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences
            y_bin (list): B/I/O classification(gold) info list
        # Return:
            Tuple with (representations and attentions if self.return_attention else None).
        """
        logits = inputs.matmul(self.attention_vector)
        unnorm_ai = (logits - logits.max()).exp()
        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = unnorm_ai.size(1)

        if y_bin is not None:
            for i, each_y_bin in enumerate(y_bin):
                inputs[i, torch.nonzero(torch.tensor(each_y_bin) == 0)] = 0

        idxes = torch.arange(
            0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        idxes = idxes.cuda()
        mask = Variable((idxes < input_lengths.unsqueeze(1)).float())

        # apply mask and renormalize attention scores (weights)
        masked_weights = unnorm_ai * mask
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums)

        # apply attention weights
        weighted = torch.mul(
            inputs, attentions.unsqueeze(-1).expand_as(inputs))
        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)
        return (representations, attentions if self.return_attention else None)


# 
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)