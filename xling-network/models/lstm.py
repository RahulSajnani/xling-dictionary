import torch

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        # or:
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout_layer = nn.Dropout(p=0.2)
		# self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0,c0))  
		out = self.drop_layer(out)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        return out

class LSTMClassifier(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, output_size, num_layers = 2):

		super(LSTMClassifier, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)

		# self.hidden2out = nn.Linear(hidden_dim, output_size)
		self.dropout_layer = nn.Dropout(p=0.2)


	def init_hidden(self, batch_size):
		return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
						autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))


	def forward(self, batch, lengths):
		
		self.hidden = self.init_hidden(batch.size(-1))

		packed_input = pack_padded_sequence(batch, lengths)
		outputs, (ht, ct) = self.lstm(packed_input, self.hidden)

		# ht is the last hidden state of the sequences
		# ht = (1 x batch_size x hidden_dim)
		# ht[-1] = (batch_size x hidden_dim)
		output = self.dropout_layer(ht[-1])
		# output = self.hidden2out(output)
		# output = self.softmax(output)

		return output

