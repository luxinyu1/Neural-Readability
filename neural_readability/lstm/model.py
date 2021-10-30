import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):

    def __init__(self, vocab_size, num_labels, device, args):

        super(LSTM, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.device = device
        self.word_embeddings = nn.Embedding(self.vocab_size, self.args.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.args.embedding_dim, self.args.hidden_dim, dropout=0.3, num_layers=5, batch_first=True)
        self.hidden2label = nn.Linear(self.args.hidden_dim, self.num_labels)

    def forward(self, input_ids, lengths):

        embedded = self.word_embeddings(input_ids)

        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        _, (h_n, c_n) = self.lstm(packed_embedded)
        
        outputs = self.hidden2label(h_n[-1])
        
        log_probs = F.log_softmax(outputs, dim=-1)

        return log_probs

class BiLSTM(nn.Module):
    
    def __init__(self, vocab_size, num_labels, device, args):

        super(BiLSTM, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.device = device
        self.word_embeddings = nn.Embedding(self.vocab_size, self.args.embedding_dim)
        self.lstm = nn.LSTM(self.args.embedding_dim, self.args.hidden_dim, dropout=0.3, bidirectional=True, num_layers=5, batch_first=True)
        self.hidden2label = nn.Linear(2 * self.args.hidden_dim, self.num_labels)

    def forward(self, input_ids, lengths):

        embedded = self.word_embeddings(input_ids) # [batch_size, batch_seq_max_length, embedding_dim]

        # Use pack_padded_sequence to avoid the "[PAD]" token influencing the input of model
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False) # let this function sort the sequence
        output, (h_n, c_n) = self.lstm(packed_embedded) # We don't need init h0 and c0, h0 and c0 defaults to zeros when not provided
        
        # h_n [num_layer*num_direction, batch_size, hidden_dim]
        
        h = torch.cat((h_n[-1], h_n[-2]), dim=-1)

        out = self.hidden2label(h)
        
        log_probs = F.log_softmax(out, dim=1)

        return log_probs