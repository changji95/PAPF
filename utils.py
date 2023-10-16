import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

PAST_LEN = 60


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def zscore(well_name):
    dyna_dir = './data/Volve production data.xlsx'
    dyna = pd.read_excel(dyna_dir, sheet_name=0, header=0)
    single_well = dyna[dyna['WELL_BORE_CODE'] == well_name]
    single_well = single_well['BORE_OIL_VOL']
    single_well = single_well.fillna(0).values.reshape([-1, 1])
    well_start = (single_well[:, 0] != 0).argmax(axis=0)
    single_well = single_well[well_start:, :]
    scaler = StandardScaler()
    scaler.fit(single_well)
    return scaler


def plot_profile(label, pred, fold=0):
    # detect folder
    figure_dir = './figure/'
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)

    WELLS_NAME = ['NO 15/9-F-1 C', 'NO 15/9-F-11 H', 'NO 15/9-F-12 H', 'NO 15/9-F-14 H', 'NO 15/9-F-15 D']
    raw_data = np.load('./data/volve_test_fd' + str(fold) + '.npy')
    history = raw_data[0, :PAST_LEN, 0]
    total_label = np.concatenate((history, label))

    # de-normalization
    scaler = zscore(WELLS_NAME[fold])
    total_label = scaler.inverse_transform(total_label.reshape(-1, 1)).flatten()
    pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

    plt.figure(figsize=[10, 5])
    len_label = len(total_label)
    plt.plot(np.arange(len_label), total_label, alpha=0.8, linewidth=1, label='Ground truth')
    plt.plot(np.arange(PAST_LEN, len_label), pred, linewidth=2, label='PAPF')
    plt.xlabel('Production days')
    plt.ylabel('Oil volume ($m^3$)')
    plt.legend()
    plt.savefig(os.path.join(figure_dir, 'PAPF_fd' + str(fold) + '.png'), dpi=300, bbox_inches='tight')
    plt.show()
