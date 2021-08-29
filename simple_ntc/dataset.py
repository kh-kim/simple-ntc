import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from torchtext.vocab import build_vocab_from_iterator


def get_vocab(texts, min_freq=1):
    vocab = build_vocab_from_iterator(
        texts,
        min_freq=min_freq,
        specials=['<PAD>', '<UNK>'],
        special_first=True
    )
    vocab.set_default_index(1)

    return vocab


class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        return {
            'text': text,
            'label': label,
        }


class TextClassificationCollator():

    def __init__(self, vocab, label_to_idx, with_text=True):
        self.vocab = vocab
        self.label_to_idx = label_to_idx
        self.with_text = with_text

    def __call__(self, samples):
        texts = [s['text'].split() for s in samples]
        encoding = [torch.tensor(self.vocab(s), dtype=torch.long) for s in texts]
        labels = [self.label_to_idx.get(s['label']) for s in samples]

        return_value = {
            'input_ids': pad_sequence(
                encoding,
                batch_first=True,
                padding_value=0,
            ),
            'labels': torch.tensor(labels, dtype=torch.long),
        }
        if self.with_text:
            return_value['text'] = texts

        return return_value
