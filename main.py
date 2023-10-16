import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model import PAPF
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--fd', default=3, type=int, help='Index of fold')
parser.add_argument('--layers', default=2, type=int, help='Number of gru layers')
parser.add_argument('--nodes', default=80, type=int, help='Number of hidden nodes')
parser.add_argument('--train', action='store_true', help='Whether to train')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAST_LEN = 60
PRED_LEN = 30
NUM_P = 2
BATCH_SIZE = 512
LR = 0.001
EPOCHS = 2000
SEED = 2023


class subDataset(Data.dataset.Dataset):
    def __init__(self, feature_1, feature_2, target):
        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        feature_1 = torch.Tensor(self.feature_1[index])
        feature_2 = torch.Tensor(self.feature_2[index])
        target = torch.Tensor(self.target[index])
        return feature_1, feature_2, target


def train(layers, nodes):
    print('Training...')
    # set the random seeds for reproducibility
    setup_seed(SEED)

    # detect ckpt folder
    ckpt_dir = './ckpt/'
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    # prepare train_loader
    train_data = np.load('./data/volve_train_fd' + str(args.fd) + '.npy')
    print('Shape of train data: ', train_data.shape)
    train_feat = torch.tensor(train_data[:, PAST_LEN:, 1:]).float()
    train_past = torch.tensor(train_data[:, :PAST_LEN, 0]).float()
    train_target = torch.tensor(train_data[:, PAST_LEN:, 0]).float()
    train_dataset = subDataset(train_feat, train_past.unsqueeze(2), train_target.unsqueeze(2))
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    # prepare val_loader
    test_data = np.load('./data/volve_test_fd' + str(args.fd) + '.npy')
    print('Shape of test data:  ', test_data.shape)
    test_feat = torch.tensor(test_data[:, PAST_LEN:, 1:]).float()
    test_past = torch.tensor(test_data[:, :PAST_LEN, 0]).float()
    test_target = torch.tensor(test_data[:, PAST_LEN:, 0]).float()
    test_dataset = subDataset(test_feat, test_past.unsqueeze(2), test_target.unsqueeze(2))
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    # training
    model = PAPF(1, NUM_P, 1, nodes, PRED_LEN, layers, device).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for i in range(EPOCHS):
        model.train()
        epoch_loss = []
        for x_feat, x_past, target in train_loader:
            x_past = x_past.to(device)
            x_feat = x_feat.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            y_pred, attn_w = model(x_past, x_feat, target, 0.5)
            single_loss = loss_function(y_pred, target)
            single_loss.backward()
            optimizer.step()
            epoch_loss.append(float(single_loss.detach().cpu()))
        if (i + 1) % 10 == 0:
            print('Epoch: %4d   Loss: %.5f' % (i + 1, np.mean(epoch_loss)))
    # save checkpoint
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'volve_fd' + str(args.fd) + '.pth'))

    # evaluating
    evaluate(model, test_loader, False)


def evaluate(model, test_loader, plot=False):
    total_true = np.zeros([0, PRED_LEN])
    total_pred = np.zeros([0, PRED_LEN])

    model.eval()
    with torch.no_grad():
        for x_feat, x_past, target in test_loader:
            x_past = x_past.to(device)
            x_feat = x_feat.to(device)
            target = target.to(device)
            y_pred, attn_w = model(x_past, x_feat, target, 0)
            total_true = np.concatenate((total_true, target.squeeze().detach().cpu().numpy()), axis=0)
            total_pred = np.concatenate((total_pred, y_pred.squeeze().detach().cpu().numpy()), axis=0)

    mse = mean_squared_error(total_true, total_pred)
    mae = mean_absolute_error(total_true, total_pred)
    r2 = r2_score(total_true, total_pred)
    print('Test MSE: %.5f' % mse)
    print('Test MAE: %.5f' % mae)
    print('Test R2 : %.5f' % r2)

    if plot:
        plot_profile(total_true.flatten(), total_pred.flatten(), args.fd)


def test(layers, nodes):
    print('Testing...')
    # prepare test_loader
    test_data = np.load('./data/volve_test_fd' + str(args.fd) + '.npy')
    print('Shape of test data:  ', test_data.shape)
    test_feat = torch.tensor(test_data[:, PAST_LEN:, 1:]).float()
    test_past = torch.tensor(test_data[:, :PAST_LEN, 0]).float()
    test_target = torch.tensor(test_data[:, PAST_LEN:, 0]).float()
    test_dataset = subDataset(test_feat, test_past.unsqueeze(2), test_target.unsqueeze(2))
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    # testing
    model = PAPF(1, NUM_P, 1, nodes, PRED_LEN, layers, device).to(device)
    ckpt_dir = './ckpt/volve_fd' + str(args.fd) + '.pth'
    model.load_state_dict(torch.load(ckpt_dir, map_location='cpu'))
    evaluate(model, test_loader, True)


if __name__ == '__main__':
    if args.train:
        train(args.layers, args.nodes)
    else:
        test(args.layers, args.nodes)
