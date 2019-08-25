import argparse

import torch
import torch.nn as nn

from data_loader import DataLoader

from simple_ntc.rnn import RNNClassifier
from simple_ntc.cnn import CNNClassifier
from simple_ntc.trainer_with_ignite import Trainer


def define_argparser():
    '''
    Define argument parser to set hyper-parameters.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--valid', required=True)
    
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--min_vocab_freq', type=int, default=2)
    p.add_argument('--max_vocab_size', type=int, default=999999)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--word_vec_size', type=int, default= 256)
    p.add_argument('--dropout', type=float, default=.3)
    
    p.add_argument('--rnn', action='store_true')
    p.add_argument('--hidden_size', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=4)

    p.add_argument('--cnn', action='store_true')
    p.add_argument('--use_batch_norm', action='store_true')
    p.add_argument('--window_sizes', type=str, default='3,4,5')
    p.add_argument('--n_filters', type=str, default='100,100,100')

    config = p.parse_args()

    config.window_sizes = list(map(int, config.window_sizes.split(',')))
    config.n_filters = list(map(int, config.n_filters.split(',')))

    return config


def main(config):
    dataset = DataLoader(train_fn=config.train,
                         valid_fn=config.valid,
                         batch_size=config.batch_size,
                         min_freq=config.min_vocab_freq,
                         max_vocab=config.max_vocab_size,
                         device=config.gpu_id
                         )

    vocab_size = len(dataset.text.vocab)
    n_classes = len(dataset.label.vocab)
    print('|vocab| =', vocab_size, '|classes| =', n_classes)

    if config.rnn is False and config.cnn is False:
        raise Exception('You need to specify an architecture to train. (--rnn or --cnn)')

    if config.rnn:
        # Declare model and loss.
        model = RNNClassifier(input_size=vocab_size,
                              word_vec_dim=config.word_vec_size,
                              hidden_size=config.hidden_size,
                              n_classes=n_classes,
                              n_layers=config.n_layers,
                              dropout_p=config.dropout
                              )
        crit = nn.NLLLoss()
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        rnn_trainer = Trainer(config)
        rnn_model = rnn_trainer.train(model, crit, dataset.train_iter, dataset.valid_iter)
    if config.cnn:
        # Declare model and loss.
        model = CNNClassifier(input_size=vocab_size,
                              word_vec_dim=config.word_vec_size,
                              n_classes=n_classes,
                              use_batch_norm=config.use_batch_norm,
                              dropout_p=config.dropout,
                              window_sizes=config.window_sizes,
                              n_filters=config.n_filters
                              )
        crit = nn.NLLLoss()
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        cnn_trainer = Trainer(config)
        cnn_model = cnn_trainer.train(model, crit, dataset.train_iter, dataset.valid_iter)

    torch.save({'rnn': rnn_model if config.rnn else None,
                'cnn': cnn_model if config.cnn else None,
                'config': config,
                'vocab': dataset.text.vocab,
                'classes': dataset.label.vocab
                }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    main(config)
