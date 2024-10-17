import pickle
import numpy as np
import pandas as pd
import os









if __name__ == '__main__':


    with open('data/data_adni_patients.pickle', 'rb') as f:
        data_dic_patients = pickle.load(f)

    data_x = data_dic_patients['data_x']
    data_y = data_dic_patients['data_y']

    # mean, 0.25, 0.75, min, max of length
    length = np.array([len(x) for x in data_x])
    print('mean:', np.mean(length))
    print('0.25:', np.percentile(length, 25))
    print('0.75:', np.percentile(length, 75))
    print('min:', np.min(length))
    print('max:', np.max(length))


    pass


