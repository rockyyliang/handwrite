import numpy as np
import os
import time
import argparse

import torch
from torch.utils.data import DataLoader

from modules import WRITER, mdn_loss_gaussian, mdn_loss_bernoulli
from sampler import SeriesSampler
from helpers import date_string, save_train_history

def main():
    '''constants'''
    SEQUENCE_LEN = args.seqlen
    BATCH_SIZE = args.batchsize
    EPOCH = args.epoch
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')

    '''load data'''
    strokes = np.load('./data/strokes-py3.npy', allow_pickle=True)
    with open('./data/sentences.txt') as f:
        texts = f.readlines()

    '''setup dataloader'''
    sampler_train = SeriesSampler(texts, strokes, sequence_len=SEQUENCE_LEN)
    dataloader_train = DataLoader(sampler_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    sampler_val = SeriesSampler(texts, strokes, sequence_len=SEQUENCE_LEN, val=True)
    dataloader_val = DataLoader(sampler_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)

    '''setup model'''
    model = WRITER().double()
    model.to(DEVICE)
    opt = torch.optim.RMSprop(model.parameters(), lr=1.0e-3)
    lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20, cooldown=10, min_lr=1e-6)

    '''create directory for saving weights'''
    weights_path = './weights'
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    save_path = os.path.join(weights_path,date_string())
    os.mkdir(save_path)
    print('saving weights to', save_path)

    '''train loop'''
    train_loss_history = []
    val_loss_history = []
    lr_history = []
    val_freq = 20
    save_freq = 500

    start = time.time()
    for e in range(EPOCH):

        iter_val = dataloader_val.__iter__()
        for b, batch in enumerate(dataloader_train):
            X = batch[0].double()
            y = batch[1].double()
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            pos, end, hidden_states = model(X)

            pos_loss = mdn_loss_gaussian(*pos, y[:,1:])
            eos_loss = mdn_loss_bernoulli(*end, y[:,0], pos_weight=1*torch.ones(1).to(DEVICE))
            loss_train = pos_loss + eos_loss

            loss_train.backward()
            opt.step()
            opt.zero_grad()

            if (b%save_freq == 0):
                #save weights
                if b!=0 or e!=0:
                    torch.save(model.state_dict(), os.path.join(save_path,'e{}b{}.pt'.format(e,b)))


            if (b%val_freq == 0):
                periodic_event = True
            else:
                periodic_event = False


            if periodic_event:

                '''VALIDATION'''
                model.eval()
                with torch.no_grad():
                    X_val, y_val = next(iter_val,1)
                    X_val = X_val.double()
                    y_val = y_val.double()
                    X_val = X_val.to(DEVICE)
                    y_val = y_val.to(DEVICE)

                    pos, end, hidden_states = model(X_val)
                    pos_loss_val = mdn_loss_gaussian(*pos, y_val[:,1:])
                    eos_loss_val = mdn_loss_bernoulli(*end, y_val[:,0], pos_weight=1*torch.ones(1).to(DEVICE))
                    loss_val = pos_loss_val + eos_loss_val
                print('batch {}, total loss: {:4f}, pos loss: {:4f}, eos loss: {:4f}, val loss: {:4f}'.format(
                    b,
                    loss_train,
                    pos_loss,
                    eos_loss,
                    loss_val
                ))

                train_loss_history.append(loss_train.item())
                val_loss_history.append(loss_val.item())
                model.train()
                '''VALIDATION OVER'''
                lr_plateau.step(loss_val)
                lr_history.append(opt.param_groups[0]['lr'])

                #if b >= 150:
                    #break

    total_time = time.time() - start
    print('Training took {:.2f} seconds'.format(total_time))

    torch.save(model.state_dict(), os.path.join(save_path,'final.pt'))

    fig_path = os.path.join(save_path, 'loss.png')
    save_train_history(train_loss_history, val_loss_history, fig_path, val_freq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seqlen', nargs='?', const=1, type=int, default=150)
    parser.add_argument('-b', '--batchsize', nargs='?', const=1, type=int, default=512)
    parser.add_argument('-e', '--epoch', nargs='?', const=1, type=int, default=1)
    args = parser.parse_args()

    main()
