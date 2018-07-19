import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMTagger(nn.Module):

    def __init__(self, pretrained_embedding, input_size, hidden_size, num_layers, num_classes):
        super(LSTMTagger, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=True)    # this counts as nn.Parameters
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.classifer = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, sent_words, sent_lens):
        embeds = self.embedding(sent_words)
        # sort for RNN
        sorted_lens, sorted_indices = torch.sort(sent_lens, dim=0, descending=True)
        sorted_embeds = torch.index_select(embeds, 0, sorted_indices)
        packed_embeds = pack_padded_sequence(sorted_embeds, sorted_lens, batch_first=True)

        # run RNN
        packed_lstm_out, _ = self.bilstm(packed_embeds)
        sorted_lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)

        # unsort
        _, unsorted_indices = torch.sort(sorted_indices, dim=0, descending=False)
        lstm_out = torch.index_select(sorted_lstm_out, 0, unsorted_indices)

        # classification
        tag_space = self.classifer(lstm_out)
        scores = F.log_softmax(tag_space, dim=2)
        scores = torch.transpose(scores, 2, 1)

        return scores

