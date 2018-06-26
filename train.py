import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import DataLoader

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('-model', required = True)
    p.add_argument('-train', required = True)
    p.add_argument('-valid', required = True)
    p.add_argument('-gpu_id', type = int, default = -1)

    p.add_argument('-batch_size', type = int, default = 64)
    p.add_argument('-n_epochs', type = int, default = 40)
    p.add_argument('-print_every', type = int, default = 50)
    p.add_argument('-early_stop', type = int, default = -1)

    p.add_argument('-max_length', type = int, default = 80)
    p.add_argument('-dropout', type = float, default = .3)
    p.add_argument('-word_vec_dim', type = int, default = 256)
    p.add_argument('-hidden_size', type = int, default = 512)

    p.add_argument('-rnn', action = 'store_true')
    p.add_argument('-n_layers', type = int, default = 4)

    p.add_argument('-cnn', action = 'store_true')
    p.add_argument('-window_size', type = str, default = '3,4,5')
    p.add_argument('-n_channels', type = str, default = '5,5,5')

    config = p.parse_args()

    config.window_size = list(map(int, config.window_size.split(',')))
    config.n_channels = list(map(int, config.n_channels.split(',')))

    return config

if __name__ == "__main__":
    config = define_argparser()

    dataset = DataLoader(train_fn = config.train, valid_fn = config.valid,
                                    batch_size = config.batch_size, 
                                    device = config.gpu_id,
                                    fix_length = config.max_length,
                                    use_eos = False,
                                    )
