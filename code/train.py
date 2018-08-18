import argparse
import logging
import os
import random
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import DataLoader
from torchnet.meter import ConfusionMeter

from data_prep.yelp_dataset import get_yelp_datasets
from data_prep.chn_hotel_dataset import get_chn_htl_datasets
from models import *
from options import opt
from vocab import Vocab
import utils

random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)

# save logs
if not os.path.exists(opt.model_save_file):
    os.makedirs(opt.model_save_file)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.model_save_file, 'log.txt'))
log.addHandler(fh)
 
# output options
log.info('Training ADAN with options:')
log.info(opt)
 

def train(opt):
    # vocab
    log.info(f'Loading Embeddings...')
    vocab = Vocab(opt.emb_filename)
    # datasets
    log.info(f'Loading data...')
    yelp_X_train = os.path.join(opt.src_data_dir, 'X_train.txt.tok.shuf.lower')
    yelp_Y_train = os.path.join(opt.src_data_dir, 'Y_train.txt.shuf')
    yelp_X_test = os.path.join(opt.src_data_dir, 'X_test.txt.tok.lower')
    yelp_Y_test = os.path.join(opt.src_data_dir, 'Y_test.txt')
    yelp_train, yelp_valid = get_yelp_datasets(vocab, yelp_X_train, yelp_Y_train,
            opt.en_train_lines, yelp_X_test, yelp_Y_test, opt.max_seq_len)
    chn_X_file = os.path.join(opt.tgt_data_dir, 'X.sent.txt.shuf.lower')
    chn_Y_file = os.path.join(opt.tgt_data_dir, 'Y.txt.shuf')
    chn_train, chn_valid, chn_test = get_chn_htl_datasets(vocab, chn_X_file, chn_Y_file,
            opt.ch_train_lines, opt.max_seq_len)
    log.info('Done loading datasets.')
    opt.num_labels = yelp_train.num_labels

    if opt.max_seq_len <= 0:
        # set to true max_seq_len in the datasets
        opt.max_seq_len = max(yelp_train.get_max_seq_len(),
                              chn_train.get_max_seq_len())
    # dataset loaders
    my_collate = utils.sorted_collate if opt.model=='lstm' else utils.unsorted_collate
    yelp_train_loader = DataLoader(yelp_train, opt.batch_size,
            shuffle=True, collate_fn=my_collate)
    yelp_train_loader_Q = DataLoader(yelp_train,
                                     opt.batch_size,
                                     shuffle=True, collate_fn=my_collate)
    chn_train_loader = DataLoader(chn_train, opt.batch_size,
            shuffle=True, collate_fn=my_collate)
    chn_train_loader_Q = DataLoader(chn_train,
                                    opt.batch_size,
                                    shuffle=True, collate_fn=my_collate)
    yelp_train_iter_Q = iter(yelp_train_loader_Q)
    chn_train_iter = iter(chn_train_loader)
    chn_train_iter_Q = iter(chn_train_loader_Q)

    yelp_valid_loader = DataLoader(yelp_valid, opt.batch_size,
            shuffle=False, collate_fn=my_collate)
    chn_valid_loader = DataLoader(chn_valid, opt.batch_size,
            shuffle=False, collate_fn=my_collate)
    chn_test_loader = DataLoader(chn_test, opt.batch_size,
            shuffle=False, collate_fn=my_collate)

    # models
    if opt.model.lower() == 'dan':
        F = DANFeatureExtractor(vocab, opt.F_layers, opt.hidden_size, opt.dropout, opt.F_bn)
    elif opt.model.lower() == 'lstm':
        F = LSTMFeatureExtractor(vocab, opt.F_layers, opt.hidden_size, opt.dropout,
                opt.bdrnn, opt.attn)
    elif opt.model.lower() == 'cnn':
        F = CNNFeatureExtractor(vocab, opt.F_layers,
                opt.hidden_size, opt.kernel_num, opt.kernel_sizes, opt.dropout)
    else:
        raise Exception('Unknown model')
    P = SentimentClassifier(opt.P_layers, opt.hidden_size, opt.num_labels,
            opt.dropout, opt.P_bn)
    Q = LanguageDetector(opt.Q_layers, opt.hidden_size, opt.dropout, opt.Q_bn)
    F, P, Q = F.to(opt.device), P.to(opt.device), Q.to(opt.device)
    optimizer = optim.Adam(list(F.parameters()) + list(P.parameters()),
                           lr=opt.learning_rate)
    optimizerQ = optim.Adam(Q.parameters(), lr=opt.Q_learning_rate)

    # training
    best_acc = 0.0
    for epoch in range(opt.max_epoch):
        F.train()
        P.train()
        Q.train()
        yelp_train_iter = iter(yelp_train_loader)
        # training accuracy
        correct, total = 0, 0
        sum_en_q, sum_ch_q = (0, 0.0), (0, 0.0)
        grad_norm_p, grad_norm_q = (0, 0.0), (0, 0.0)
        for i, (inputs_en, targets_en) in tqdm(enumerate(yelp_train_iter),
                                               total=len(yelp_train)//opt.batch_size):
            try:
                inputs_ch, _ = next(chn_train_iter)  # Chinese labels are not used
            except:
                # check if Chinese data is exhausted
                chn_train_iter = iter(chn_train_loader)
                inputs_ch, _ = next(chn_train_iter)

            # Q iterations
            n_critic = opt.n_critic
            if n_critic>0 and ((epoch==0 and i<=25) or (i%500==0)):
                n_critic = 10
            utils.freeze_net(F)
            utils.freeze_net(P)
            utils.unfreeze_net(Q)
            for qiter in range(n_critic):
                # clip Q weights
                for p in Q.parameters():
                    p.data.clamp_(opt.clip_lower, opt.clip_upper)
                Q.zero_grad()
                # get a minibatch of data
                try:
                    # labels are not used
                    q_inputs_en, _ = next(yelp_train_iter_Q)
                except StopIteration:
                    # check if dataloader is exhausted
                    yelp_train_iter_Q = iter(yelp_train_loader_Q)
                    q_inputs_en, _ = next(yelp_train_iter_Q)
                try:
                    q_inputs_ch, _ = next(chn_train_iter_Q)
                except StopIteration:
                    chn_train_iter_Q = iter(chn_train_loader_Q)
                    q_inputs_ch, _ = next(chn_train_iter_Q)

                features_en = F(q_inputs_en)
                o_en_ad = Q(features_en)
                l_en_ad = torch.mean(o_en_ad)
                (-l_en_ad).backward()
                log.debug(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
                sum_en_q = (sum_en_q[0] + 1, sum_en_q[1] + l_en_ad.item())

                features_ch = F(q_inputs_ch)
                o_ch_ad = Q(features_ch)
                l_ch_ad = torch.mean(o_ch_ad)
                l_ch_ad.backward()
                log.debug(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
                sum_ch_q = (sum_ch_q[0] + 1, sum_ch_q[1] + l_ch_ad.item())

                optimizerQ.step()

            # F&P iteration
            utils.unfreeze_net(F)
            utils.unfreeze_net(P)
            utils.freeze_net(Q)
            if opt.fix_emb:
                utils.freeze_net(F.word_emb)
            # clip Q weights
            for p in Q.parameters():
                p.data.clamp_(opt.clip_lower, opt.clip_upper)
            F.zero_grad()
            P.zero_grad()
            
            features_en = F(inputs_en)
            o_en_sent = P(features_en)
            l_en_sent = functional.nll_loss(o_en_sent, targets_en)
            l_en_sent.backward(retain_graph=True)
            o_en_ad = Q(features_en)
            l_en_ad = torch.mean(o_en_ad)
            (opt.lambd*l_en_ad).backward(retain_graph=True)
            # training accuracy
            _, pred = torch.max(o_en_sent, 1)
            total += targets_en.size(0)
            correct += (pred == targets_en).sum().item()

            features_ch = F(inputs_ch)
            o_ch_ad = Q(features_ch)
            l_ch_ad = torch.mean(o_ch_ad)
            (-opt.lambd*l_ch_ad).backward()

            optimizer.step()
    
        # end of epoch
        log.info('Ending epoch {}'.format(epoch+1))
        # logs
        if sum_en_q[0] > 0:
            log.info(f'Average English Q output: {sum_en_q[1]/sum_en_q[0]}')
            log.info(f'Average Foreign Q output: {sum_ch_q[1]/sum_ch_q[0]}')
        # evaluate
        log.info('Training Accuracy: {}%'.format(100.0*correct/total))
        log.info('Evaluating English Validation set:')
        evaluate(opt, yelp_valid_loader, F, P)
        log.info('Evaluating Foreign validation set:')
        acc = evaluate(opt, chn_valid_loader, F, P)
        if acc > best_acc:
            log.info(f'New Best Foreign validation accuracy: {acc}')
            best_acc = acc
            torch.save(F.state_dict(),
                    '{}/netF_epoch_{}.pth'.format(opt.model_save_file, epoch))
            torch.save(P.state_dict(),
                    '{}/netP_epoch_{}.pth'.format(opt.model_save_file, epoch))
            torch.save(Q.state_dict(),
                    '{}/netQ_epoch_{}.pth'.format(opt.model_save_file, epoch))
        log.info('Evaluating Foreign test set:')
        evaluate(opt, chn_test_loader, F, P)
    log.info(f'Best Foreign validation accuracy: {best_acc}')


def evaluate(opt, loader, F, P):
    F.eval()
    P.eval()
    it = iter(loader)
    correct = 0
    total = 0
    confusion = ConfusionMeter(opt.num_labels)
    with torch.no_grad():
        for inputs, targets in tqdm(it):
            outputs = P(F(inputs))
            _, pred = torch.max(outputs, 1)
            confusion.add(pred.data, targets.data)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
    accuracy = correct / total
    log.info('Accuracy on {} samples: {}%'.format(total, 100.0*accuracy))
    log.debug(confusion.conf)
    return accuracy


if __name__ == '__main__':
    train(opt)
