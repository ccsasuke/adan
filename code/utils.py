import pdb
import numpy as np
import torch
from torch.utils.serialization import load_lua
from options import opt

def freeze_net(net):
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    for p in net.parameters():
        p.requires_grad = True


def sorted_collate(batch):
    return my_collate(batch, sort=True)


def unsorted_collate(batch):
    return my_collate(batch, sort=False)


def my_collate(batch, sort):
    x, y = zip(*batch)
    x, y = pad(x, y, opt.eos_idx, sort)
    x = (x[0].to(opt.device), x[1].to(opt.device))
    y = y.to(opt.device)
    return (x, y)


def pad(x, y, eos_idx, sort):
    inputs, lengths = zip(*x)
    max_len = max(lengths)
    # pad sequences
    padded_inputs = torch.full((len(inputs), max_len), eos_idx, dtype=torch.long)
    for i, row in enumerate(inputs):
        assert eos_idx not in row, f'EOS in sequence {row}'
        padded_inputs[i][:len(row)] = torch.tensor(row, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long).view(-1)
    if sort:
        # sort by length
        sort_len, sort_idx = lengths.sort(0, descending=True)
        padded_inputs = padded_inputs.index_select(0, sort_idx)
        y = y.index_select(0, sort_idx)
        return (padded_inputs, sort_len), y
    else:
        return (padded_inputs, lengths), y


def zero_eos(emb, eos_idx):
    emb.weight.data[eos_idx].zero_()
