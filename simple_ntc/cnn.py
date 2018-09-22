import torch
import torch.nn as nn


class CNNClassifier(nn.Module):

    def __init__(self,
                 input_size,
                 word_vec_dim,
                 n_classes,
                 dropout_p=.2,
                 window_sizes=[3, 4, 5],
                 n_filters=[10, 10, 10]
                 ):
        self.input_size = input_size
        self.word_vec_dim = word_vec_dim
        self.n_classes = n_classes
        self.dropout_p = dropout_p
        self.window_sizes = window_sizes
        self.n_filters = n_filters

        super().__init__()

        self.emb = nn.Embedding(input_size, word_vec_dim)
        for window_size, n_filter in zip(window_sizes, n_filters):
            cnn = nn.Conv2d(in_channels=1,
                            out_channels=n_filter,
                            kernel_size=(window_size, word_vec_dim)
                            )
            setattr(self, 'cnn-%d-%d' % (window_size, n_filter), cnn)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.generator = nn.Linear(sum(n_filters), n_classes)
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_dim)
        min_length = max(self.window_sizes)
        if min_length > x.size(1):
            pad = x.new(x.size(0), min_length - x.size(1), self.word_vec_dim).zero_()
            # |pad| = (batch_size, min_length - length, word_vec_dim)
            x = torch.cat([x, pad], dim=1)
            # |x| = (batch_size, min_length, word_vec_dim)
        
        x = x.unsqueeze(1)
        # |x| = (batch_size, 1, length, word_vec_dim)

        cnn_outs = []
        for window_size, n_filter in zip(self.window_sizes, self.n_filters):
            cnn = getattr(self, 'cnn-%d-%d' % (window_size, n_filter))
            cnn_out = self.dropout(self.relu(cnn(x)))
            # |x| = (batch_size, n_filter, length - window_size + 1, 1)
            cnn_out = nn.functional.max_pool1d(input=cnn_out.squeeze(-1),
                                            kernel_size=cnn_out.size(-2)
                                            ).squeeze(-1)
            # |cnn_out| = (batch_size, n_filter)
            cnn_outs += [cnn_out]
        cnn_outs = torch.cat(cnn_outs, dim=-1)
        # |cnn_outs| = (batch_size, sum(n_filters))
        y = self.activation(self.generator(cnn_outs))
        # |y| = (batch_size, n_classes)

        return y
