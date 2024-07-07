# conda activate ts-prompt
"""
prefix tuning to incorporate multiple variables

https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
https://github.com/allenai/better-promptability/blob/5cb1e33c9988f6f973e92a1c78da8291fe55df64/better_promptability/models/t5_with_prefix.py



- check how many parameters
    - 4310MiB vs 7120MiB for imputation

- classify and anomaly only 1 channel
    - do forecasting too


TODO:
- does it make sense to flatten the data to train deep prompt?
    - time dimension gets flattened too (#patches)
- should I exclude the MPT when training the deep prompt?


- need one deep prompt for each dataset, if #channels is different
    - what can you do if #channels is different?
    - just add them up?
    - train one layer for each dataset, project into the same space
    - lora-like: u_i * v
    - RNN, flip channel and time dimension
- solution: flip channel and time dimension, subsample time dimension to the same size (maybe 16)
    - do attention block




- did flattening and not flattening give similar performance? why?

- can we draw semantic meaning from the prompt?



- now only works with 1 batch size
- classify test data strangely long
- run experiments with original moment, single task, finetune
"""

"""
experiments:
1. zero-shot
2. finetune
    - all tasks
    - (one for each task probably tasks a long time)
        - TODO: should compare to this, since this is the use case
3. prompt tuning
    - multitask only
    - multi variable only
    - multitask and multi variable

"""

import torch.cuda.amp


import os
import torch

from momentfm.utils.utils import control_randomness

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 0


# Set random seeds for PyTorch, Numpy etc.
control_randomness(seed=RANDOM_SEED)





if __name__ == "__main__":

    from data import get_data
    from experiments import zero_shot, finetune, prompt_tuning


    # os.environ["WANDB_MODE"] = "offline"

    EXPERIMENT_NAME = 'finetune'

    multitask = True
    multivariable = True

    dataset_names = ['imputation', 'anomaly', 'classify', 'forecasting_long']

    # if not multitask and not multivariable and EXPERIMENT_NAME == 'prompt_tuning':
    #     dataset_names = ['forecasting_long']

    batch_size = 8

    name = ''
    if EXPERIMENT_NAME == 'zero_shot':
        experiment = zero_shot
    elif EXPERIMENT_NAME == 'finetune':
        experiment = finetune
    elif EXPERIMENT_NAME == 'prompt_tuning':
        experiment = prompt_tuning
        name = f'_multitask_{multitask}_multivariable_{multivariable}'
    else:
        raise NotImplementedError


    train_loader, test_loader = get_data(batch_size=batch_size, dataset_names=dataset_names)

    model = experiment(train_loader, test_loader, name,
                       prefix_tuning_multi=multivariable,
                       MPT=multitask,
                       )
