from abc import ABC
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from dpgc import device


# TODO: add base class for lstm model
class PGCLSTM(ABC, nn.Module):
    pass

class BiPGCLSTM(nn.Module):
    def __init__(self,
                 phonemes2index: dict,
                 graphemes2index: dict,
                 pad_token: str = "<PAD>",
                 num_layers_lstm: int = 1,
                 num_lstm_units = 100,
                 embedding_dim = 50,
                 batch_size=3):
        super(BiPGCLSTM, self).__init__()
        self.phonemes = phonemes2index
        self.graphemes = graphemes2index
        self.pad_token = pad_token
        self.num_layers_lstm = num_layers_lstm
        self.num_lstm_units = num_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        # don't count the padding tag for the classifier output
        self.num_graphemes = len(self.graphemes)

        # when the model is bidirectional we double the output dimension
        self.lstm = None

        # build actual NN
        self.__build_model()

    def __build_model(self):
        # build embedding layer first
        nb_vocab_words = len(self.phonemes)

        # whenever the embedding sees the padding index it'll make the whole vector zeros
        padding_idx = self.phonemes[self.pad_token]
        self.word_embedding = nn.Embedding(
            num_embeddings=nb_vocab_words,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx
        )

        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.num_lstm_units,
            num_layers=self.num_layers_lstm,
            batch_first=True,
            bidirectional=True
        )

        # output layer which projects back to tag space (2 x because of bidirectional lstm)
        self.linear_out = nn.Linear(self.num_lstm_units * 2, self.num_graphemes)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.num_layers_lstm, self.batch_size, self.num_lstm_units)
        hidden_b = torch.randn(self.num_layers_lstm, self.batch_size, self.num_lstm_units)

        hidden_a = hidden_a.to(device)
        hidden_b = hidden_b.to(device)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return hidden_a, hidden_b

    def forward(self, X, X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()

        batch_size, seq_len = X.size()

        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        X = self.word_embedding(X)
        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        # now run through LSTM
        X, self.hidden = self.lstm(X)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        # run through actual linear layer
        X = self.linear_out(X)

        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        X = F.log_softmax(X, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        X = X.view(batch_size, seq_len, self.num_graphemes)

        Y_hat = X
        return Y_hat

    def loss(self, Y_hat, Y):
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.



        # flatten all the labels
        #Y = torch.nn.functional.one_hot(Y, num_classes=self.num_graphemes)
        Y = Y.view(-1)

        #logger.info(Y.shape)
        #logger.info(Y)

        #logger.info(Y_hat.shape)
        #logger.info(Y_hat)
        # flatten all predictions
        Y_hat = Y_hat.view(-1, self.num_graphemes)

        #logger.info(Y_hat.shape)
        #logger.info(Y_hat)

        ce_loss_func = torch.nn.CrossEntropyLoss()
        ce_loss = ce_loss_func(Y_hat, Y)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        #tag_pad_token = self.graphemes[self.pad_token]
        #mask = (Y > tag_pad_token).float()
        #logger.info(mask)
        #logger.info(torch.sum(mask))
        # count how many tokens we have
        #nb_tokens = int(torch.sum(mask).item())

        # pick the values for the label and zero out the rest with the mask
        #Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        #ce_loss = -torch.sum(Y_hat) / nb_tokens

        return ce_loss


if __name__ == "__main__":
    pass