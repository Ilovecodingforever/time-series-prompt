


import torch.cuda.amp


import sys
sys.path.append('/zfsauton2/home/mingzhul/time-series-prompt/src/momentfm')

import os
import torch
import datetime


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"
os.environ["WANDB_CACHE_DIR"] = "/home/scratch/mingzhul/.cache/wandb"

from momentfm.utils.utils import control_randomness

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 13


# Set random seeds for PyTorch, Numpy etc.
control_randomness(seed=RANDOM_SEED)


from data import load_mimic

import numpy as np
import pandas as pd
import pickle





seed = 0
batch_size = 1
task = 'classification'
equal_length = False
small_part = True
ordinal = True
benchmark = 'mortality'

train_loader, val_loader, test_loader = load_mimic(equal_length=equal_length, small_part=small_part,
                                                    benchmark=benchmark, ordinal=ordinal, seed=seed, batch_size=batch_size,
                                                    normalize=False)



for name, loader in {'train': train_loader, 'val': val_loader, 'test': test_loader}.items():

    lst = []
    for i, (data, mask, label) in enumerate(loader):
        # TODO: drop time?
        # TODO: only use 3 decimal places?
        data = data[0][:, mask[0].bool()]
        data = pd.DataFrame(data.numpy().T)
        # remove imputed values
        for feature in data.columns:
            data[feature] = data[feature].drop_duplicates()
            # prepend the coulmn name to values, if not nan
            data[feature] = data[feature].apply(lambda x: f"{feature}_{x}" if not np.isnan(x) else np.nan)
        # flatten the data
        data = data.to_numpy().flatten()
        data = [d for d in data if isinstance(d, str)]

        pt_id = i
        label = label.item()
        seq_list = data
        segment_list = list(np.ones(len(seq_list)))

        lst.append([pt_id, label, seq_list, segment_list])


    pickle.dump(lst, open(f'/home/scratch/mingzhul/mimic_medbert_{name}.pkl', 'wb'))



