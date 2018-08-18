import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


class ChnHtlDataset(Dataset):
    def __init__(self, X, Y, num_train_lines, vocab, max_seq_len, update_vocab):
        # data is assumed to be pre-shuffled
        if num_train_lines > 0:
            X = X[:num_train_lines]
            Y = Y[:num_train_lines]
        if update_vocab:
            for x in X:
                for w in x:
                    vocab.add_word(w)
        # save lengths
        self.X = [([vocab.lookup(w) for w in x], len(x)) for x in X]
        if max_seq_len > 0:
            self.set_max_seq_len(max_seq_len)
        self.Y = Y
        self.num_labels = 5
        assert len(self.X) == len(self.Y), 'X and Y have different lengths'
        print('Loaded Chinese Hotel dataset of {} samples'.format(len(self.X)))

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

    def set_max_seq_len(self, max_seq_len):
        self.X = [(x[0][:max_seq_len], min(x[1], max_seq_len)) for x in self.X]
        self.max_seq_len = max_seq_len

    def get_max_seq_len(self):
        if not hasattr(self, 'max_seq_len'):
            self.max_seq_len = max([x[1] for x in self.X])
        return self.max_seq_len

    def get_subset(self, num_lines):
        return ChnHtlDataset(self.X[:num_lines], self.Y[:num_lines],
                             0, self.max_seq_len)


def get_chn_htl_datasets(vocab, X_filename, Y_filename, num_train_lines, max_seq_len):
    """
    dataset is pre-shuffled
    split: 150k train + 10k valid + 10k test
    """
    num_train = 150000
    num_valid = num_test = 10000
    num_total = num_train + num_valid + num_test
    raw_X = []
    with open(X_filename) as inf:
        for line in inf:
            words = line.rstrip().split()
            if max_seq_len > 0:
                words = words[:max_seq_len]
            raw_X.append(words)
    Y = (torch.from_numpy(np.loadtxt(Y_filename)) - 1).long()
    assert num_total == len(raw_X) == len(Y), 'X and Y have different lengths'

    train_dataset = ChnHtlDataset(raw_X[:num_train], Y[:num_train], num_train_lines,
            vocab, max_seq_len, update_vocab=True)
    valid_dataset = ChnHtlDataset(raw_X[num_train:num_train+num_valid],
                                  Y[num_train:num_train+num_valid],
                                  0,
                                  vocab,
                                  max_seq_len,
                                  update_vocab=False)
    test_dataset = ChnHtlDataset(raw_X[num_train+num_valid:],
                                 Y[num_train+num_valid:],
                                 0,
                                 vocab,
                                 max_seq_len,
                                 update_vocab=False)
    return train_dataset, valid_dataset, test_dataset
