from torchtext import data, datasets

class DataLoader(object):

    def __init__(self, train_fn, valid_fn, 
                    batch_size = 64, 
                    device = -1, 
                    max_vocab = 9999999, 
                    fix_length = None, 
                    use_eos = False, 
                    shuffle = True
                    ):
        super(DataLoader, self).__init__()

        self.label = data.Field(sequential = False, use_vocab = False)
        self.text = data.Field(use_vocab = True, 
                                batch_first = True, 
                                include_lengths = True, 
                                fix_length = fix_length, 
                                eos_token = '<EOS>' if use_eos else None
                                )

        train, valid = data.TabularDataset.splits(path = '', 
                                                    train = train_fn, 
                                                    validation = valid_fn, 
                                                    format = 'tsv', 
                                                    fields = [('label', self.label), ('text', self.text)]
                                                    )

        self.train_iter, self.valid_iter = data.BucketIterator.splits((train, valid), 
                                                                        batch_size = batch_size, 
                                                                        device = 'cuda:%d' % device if device >= 0 else 'cpu', 
                                                                        shuffle = shuffle
                                                                        )
        
        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size = max_vocab)
