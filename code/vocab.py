import numpy as np
import torch
import torch.nn as nn

from options import opt


class Vocab:
    def __init__(self, txt_file):
        self.vocab_size, self.emb_size = 0, opt.emb_size
        self.embeddings = []
        self.w2vvocab = {}
        self.v2wvocab = []
        # load pretrained embedings
        with open(txt_file, 'r') as inf:
            parts = inf.readline().split()
            assert len(parts) == 2
            vs, es = int(parts[0]), int(parts[1])
            assert es == self.emb_size
            # add an UNK token
            self.pretrained = np.empty((vs, es), dtype=np.float)
            self.pt_v2wvocab = []
            self.pt_w2vvocab = {}
            cnt = 0
            for line in inf:
                parts = line.rstrip().split(' ')
                word = parts[0]
                # add to vocab
                self.pt_v2wvocab.append(word)
                self.pt_w2vvocab[word] = cnt
                # load vector
                if len(parts) == 2:  # comma separated
                    vecs = parts[-1]
                    vector = [float(x) for x in vecs.split(',')]
                else:
                    vector = [float(x) for x in parts[-self.emb_size:]]
                self.pretrained[cnt] = vector
                cnt += 1
        # add <unk>
        self.unk_tok = '<unk>'
        self.add_word(self.unk_tok)
        self.unk_idx = self.w2vvocab[self.unk_tok]
        # add EOS token
        self.eos_tok = '</s>'
        self.add_word(self.eos_tok)
        opt.eos_idx = self.eos_idx = self.w2vvocab[self.eos_tok]
        self.embeddings[self.eos_idx][:] = 0

    def base_form(word):
        return word.strip().lower()

    def new_rand_emb(self):
        vec = np.random.normal(0, 1, size=self.emb_size)
        vec /= sum(x*x for x in vec) ** .5
        return vec

    def init_embed_layer(self):
        # free some memory
        self.clear_pretrained_vectors()
        emb = nn.Embedding.from_pretrained(torch.tensor(self.embeddings, dtype=torch.float),
                freeze=opt.fix_emb)
        assert len(emb.weight) == self.vocab_size
        return emb

    def add_word(self, word):
        word = Vocab.base_form(word)
        if word not in self.w2vvocab:
            if not opt.random_emb and hasattr(self, 'pt_w2vvocab'):
                if opt.fix_unk and word not in self.pt_w2vvocab:
                    # use fixed unk token, do not update vocab
                    return
                if word in self.pt_w2vvocab:
                    vector = self.pretrained[self.pt_w2vvocab[word]].copy()
                else:
                    vector = self.new_rand_emb()
            else:
                vector = self.new_rand_emb()
            self.v2wvocab.append(word)
            self.w2vvocab[word] = self.vocab_size
            self.embeddings.append(vector)
            self.vocab_size += 1

    def clear_pretrained_vectors(self):
        del self.pretrained
        del self.pt_w2vvocab
        del self.pt_v2wvocab

    def lookup(self, word):
        word = Vocab.base_form(word)
        if word in self.w2vvocab:
            return self.w2vvocab[word]
        return self.unk_idx

    def get_word(self, i):
        return self.v2wvocab[i]
