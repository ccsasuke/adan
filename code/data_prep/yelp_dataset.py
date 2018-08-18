import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


class YelpDataset(Dataset):
    def __init__(self, X_file, Y_file, num_train_lines, vocab, max_seq_len, update_vocab):
        self.raw_X = []
        self.X = []
        with open(X_file) as inf:
            cnt = 0
            for line in inf:
                words = line.rstrip().split()
                if max_seq_len > 0:
                    words = words[:max_seq_len]
                self.raw_X.append(words)
                if update_vocab:
                    for w in words:
                        vocab.add_word(w)
                # save lengths
                self.X.append(([vocab.lookup(w) for w in words], len(words)))
                cnt += 1
                if num_train_lines > 0 and cnt >= num_train_lines:
                    break

        self.max_seq_len = max_seq_len
        if isinstance(Y_file, str):
            self.Y = (torch.from_numpy(np.loadtxt(Y_file)) - 1).long()
        else:
            self.Y = Y_file
        if num_train_lines > 0:
            self.X = self.X[:num_train_lines]
            self.Y = self.Y[:num_train_lines]
        self.num_labels = 5
        # self.Y = self.Y.to(opt.device)
        assert len(self.X) == len(self.Y), 'X and Y have different lengths'
        print('Loaded Yelp dataset of {} samples'.format(len(self.X)))

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

def get_yelp_datasets(vocab,
                      X_train_filename,
                      Y_train_filename,
                      num_train_lines,
                      X_test_filename,
                      Y_test_filename,
                      max_seq_len):
    train_dataset = YelpDataset(X_train_filename, Y_train_filename,
            num_train_lines, vocab, max_seq_len, update_vocab=True)
    valid_dataset = YelpDataset(X_test_filename, Y_test_filename,
            0, vocab, max_seq_len, update_vocab=True)
    return train_dataset, valid_dataset
