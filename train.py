import argparse

import torch
import torch.nn as nn

from data_loader import DataLoader

from simple_ntc.rnn import RNNClassifier
from simple_ntc.trainer import Trainer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--valid', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--min_vocab_freq', type=int, default=2)
    p.add_argument('--max_vocab_size', type=int, default=999999)
    p.add_argument('--max_length', type=int, default=80)

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=40)
    p.add_argument('--early_stop', type=int, default=-1)

    p.add_argument('--dropout', type=float, default=.2)
    p.add_argument('--word_vec_dim', type=int, default=128)
    p.add_argument('--hidden_size', type=int, default=256)

    p.add_argument('--rnn', action='store_true')
    p.add_argument('--n_layers', type=int, default=4)

    p.add_argument('--cnn', action='store_true')
    p.add_argument('--window_size', type=str, default='3,4,5')
    p.add_argument('--n_channels', type=str, default='5,5,5')

    config = p.parse_args()

    config.window_size = list(map(int, config.window_size.split(',')))
    config.n_channels = list(map(int, config.n_channels.split(',')))

    return config


if __name__ == "__main__":
    config = define_argparser()

    dataset = DataLoader(train_fn=config.train,
                         valid_fn=config.valid,
                         batch_size=config.batch_size,
                         min_freq=config.min_vocab_freq,
                         max_vocab=config.max_vocab_size,
                         device=config.gpu_id,
                         use_eos=False,
                         )

    vocab_size = len(dataset.text.vocab)
    n_classes = len(dataset.label.vocab)
    print(vocab_size, n_classes)

    model = RNNClassifier(input_size=vocab_size,
                          word_vec_dim=config.word_vec_dim,
                          hidden_size=config.hidden_size,
                          n_classes=n_classes,
                          n_layers=config.n_layers,
                          dropout_p=config.dropout
                          )
    crit = nn.NLLLoss()

    trainer = Trainer(model, crit)
    trainer.train(dataset.train_iter,
                  dataset.valid_iter,
                  batch_size=config.batch_size,
                  n_epochs=config.n_epochs,
                  early_stop=config.early_stop,
                  verbose=config.verbose
                  )
    