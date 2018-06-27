import torch
import torch.nn as nn

class RNNClassifier(nn.Module):

    def __init__(self, word_vec_dim, hidden_size, n_classes, n_layers = 4, dropout_p = .2):
        