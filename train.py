import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from simple_ntc.trainer import Trainer
from simple_ntc.dataset import (
    VocabBuilder,
    TextClassificationDataset,
    TextClassificationCollator,
)
from simple_ntc.utils import read_text

from simple_ntc.models.rnn import RNNClassifier
from simple_ntc.models.cnn import CNNClassifier


def define_argparser():
    '''
    Define argument parser to set hyper-parameters.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    p.add_argument('--valid_ratio', type=float, default=.2)
    
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--min_vocab_freq', type=int, default=5)
    p.add_argument('--max_vocab_size', type=int, default=999999)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--word_vec_size', type=int, default= 256)
    p.add_argument('--dropout', type=float, default=.3)

    p.add_argument('--max_length', type=int, default=256)
    
    p.add_argument('--rnn', action='store_true')
    p.add_argument('--hidden_size', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=4)

    p.add_argument('--cnn', action='store_true')
    p.add_argument('--use_batch_norm', action='store_true')
    p.add_argument('--window_sizes', type=int, nargs='*', default=[3, 4, 5])
    p.add_argument('--n_filters', type=int, nargs='*', default=[100, 100, 100])

    config = p.parse_args()

    return config


def get_loaders(fn, valid_ratio=.2):
    # Get list of labels and list of texts.
    labels, texts = read_text(fn)

    # Generate label to index map.
    unique_labels = list(set(labels))
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i
        index_to_label[i] = label

    # Shuffle before split into train and validation set.
    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(texts) * (1 - valid_ratio))

    vocab = VocabBuilder(texts[:idx]).get_vocab(
        min_freq=config.min_vocab_freq
    )

    # Get dataloaders using given tokenizer as collate_fn.
    train_loader = DataLoader(
        TextClassificationDataset(texts[:idx], labels[:idx]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TextClassificationCollator(vocab, label_to_index, config.max_length),
    )
    valid_loader = DataLoader(
        TextClassificationDataset(texts[idx:], labels[idx:]),
        batch_size=config.batch_size,
        collate_fn=TextClassificationCollator(vocab, label_to_index, config.max_length),
    )

    return train_loader, valid_loader, vocab, index_to_label


def main(config):
    # Get dataloaders, vocabulary and index_to_label map.
    train_loader, valid_loader, vocab, index_to_label = get_loaders(
        config.train_fn,
        valid_ratio=config.valid_ratio
    )

    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
    )

    vocab_size = len(vocab)
    n_classes = len(index_to_label)
    print('|vocab| =', vocab_size, '|classes| =', n_classes)

    if config.rnn is False and config.cnn is False:
        raise Exception('You need to specify an architecture to train. (--rnn or --cnn)')

    if config.rnn:
        # Declare model and loss.
        model = RNNClassifier(
            input_size=vocab_size,
            word_vec_size=config.word_vec_size,
            hidden_size=config.hidden_size,
            n_classes=n_classes,
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        )
        optimizer = optim.Adam(model.parameters())
        crit = nn.NLLLoss()
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        rnn_trainer = Trainer(config)
        rnn_model = rnn_trainer.train(
            model,
            crit,
            optimizer,
            train_loader,
            valid_loader
        )
    if config.cnn:
        # Declare model and loss.
        model = CNNClassifier(
            input_size=vocab_size,
            word_vec_size=config.word_vec_size,
            n_classes=n_classes,
            use_batch_norm=config.use_batch_norm,
            dropout_p=config.dropout,
            window_sizes=config.window_sizes,
            n_filters=config.n_filters,
        )
        optimizer = optim.Adam(model.parameters())
        crit = nn.NLLLoss()
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        cnn_trainer = Trainer(config)
        cnn_model = cnn_trainer.train(
            model,
            crit,
            optimizer,
            train_loader,
            valid_loader
        )

    torch.save({
        'rnn': rnn_model.state_dict() if config.rnn else None,
        'cnn': cnn_model.state_dict() if config.cnn else None,
        'config': config,
        'vocab': vocab,
        'classes': index_to_label,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
