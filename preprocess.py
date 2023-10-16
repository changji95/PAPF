import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def prep_volve_data(well_name, past_len, pred_len, stride, test_index, train=True):
    dyna_dir = './data/Volve production data.xlsx'
    dyna = pd.read_excel(dyna_dir, sheet_name=0, header=0)

    sample_len = past_len + pred_len
    num_p = len(PHY)
    total_samples = np.zeros((0, sample_len, num_p + 1))

    for name in well_name:
        print('Process wellbore: %s' % name)
        single_well = dyna[dyna['WELL_BORE_CODE'] == name]
        single_well = single_well[TGT + PHY]
        # filling with zero
        single_well = single_well.fillna(0).values
        well_start = (single_well[:, 0] != 0).argmax(axis=0)
        single_well = single_well[well_start:, :]
        len_well = len(single_well)

        # z-score normalization
        scaler = StandardScaler()
        single_well_n = scaler.fit_transform(single_well)

        # sample generation
        num_samples = (len_well - sample_len) // stride + 1
        samples = np.zeros((num_samples, sample_len, num_p + 1))
        for j in range(num_samples):
            samples[j, :, :] = single_well_n[j * stride:j * stride + sample_len, :]
        total_samples = np.concatenate((total_samples, samples), axis=0)

    prefix = 'train' if train else 'test'
    np.save(os.path.join('data', 'volve_' + prefix + '_fd' + str(test_index)), total_samples)
    print('Shape of ' + prefix + ' data: ', total_samples.shape)


if __name__ == '__main__':
    WELL_NAMES = ['NO 15/9-F-1 C', 'NO 15/9-F-11 H', 'NO 15/9-F-12 H', 'NO 15/9-F-14 H', 'NO 15/9-F-15 D']
    PHY = ['AVG_CHOKE_SIZE_P', 'AVG_WHP_P']
    TGT = ['BORE_OIL_VOL']
    PAST_LEN = 60
    PRED_LEN = 30
    TRAIN_STRIDE = 10

    test_stride = PRED_LEN

    # data partition
    print('Preprocessing...')
    for i in range(5):
        print('\nFold %d' % i)
        train_well = WELL_NAMES.copy()
        test_well = [WELL_NAMES[i]]
        train_well.pop(i)
        prep_volve_data(train_well, PAST_LEN, PRED_LEN, TRAIN_STRIDE, i, True)
        prep_volve_data(test_well, PAST_LEN, PRED_LEN, test_stride, i, False)
