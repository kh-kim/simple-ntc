import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from simple_ntc.bert_trainer import BertTrainer as Trainer
from simple_ntc.data_loader import BertDataset, TokenizerWrapper


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    p.add_argument('--bert_name', type=str, default='beomi/kcbert-base')
    
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup_ratio', type=float, default=.1)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)

    p.add_argument('--max_length', type=int, default=100)

    config = p.parse_args()

    return config


def read_text(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()

        labels, texts = [], []
        for line in lines:
            if line.strip() != '':
                label, text = line.strip().split('\t')
                labels += [label]
                texts += [text]

    return labels, texts


def get_loaders(fn, tokenizer):
    labels, texts = read_text(fn)

    unique_labels = list(set(labels))
    label_to_index = {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i

    labels = list(map(label_to_index.get, labels))

    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(texts) * .8)

    train_loader = DataLoader(
        BertDataset(texts[:idx], labels[:idx]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TokenizerWrapper(tokenizer, config.max_length).collate,
    )
    valid_loader = DataLoader(
        BertDataset(texts[idx:], labels[idx:]),
        batch_size=config.batch_size,
        collate_fn=TokenizerWrapper(tokenizer, config.max_length).collate,
    )

    return train_loader, valid_loader, label_to_index


def main(config):
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name)
    train_loader, valid_loader, label_to_index = get_loaders(config.train_fn, tokenizer)

    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
    )

    model = BertForSequenceClassification.from_pretrained(
        config.bert_name,
        num_labels=len(label_to_index)
    )
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.lr,
        eps=config.adam_epsilon
    )
    crit = nn.CrossEntropyLoss()

    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    trainer = Trainer(config)
    model = trainer.train(
        model,
        crit,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
    )

    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': label_to_index,
        'tokenizer': tokenizer,
    }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    main(config)
