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


- use all data

- what's the common argument in literature for prompt tuning (saving params)?
    - https://arxiv.org/pdf/2110.07602



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


import sys
sys.path.append('/zfsauton2/home/mingzhul/time-series-prompt/src/momentfm')

import os
import torch
import datetime


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"
os.environ["WANDB_CACHE_DIR"] = "/home/scratch/mingzhul/.cache/wandb"

from momentfm.utils.utils import control_randomness

RANDOM_SEED = 13


# Set random seeds for PyTorch, Numpy etc.
control_randomness(seed=RANDOM_SEED)





def classify_experiments(experiment_name: str,
                         model_name: str,
                         multivariate_projection: str,
                         agg: str,
                         epochs: int = 10,
                         save_model: bool = False,
                         bootstrap: bool = False,
                         num_prefix: int = 16,
                         flatten: bool = False):

    from data import get_data
    from experiments import finetune, prompt_tuning, lora, linearprobe


    task = 'classification'

    batch_size = 1

    name = model_name + '/'
    if 'finetune_prompt' in experiment_name:
        experiment = prompt_tuning
        name += experiment_name
    elif 'finetune' in experiment_name:
        experiment = finetune
        name += experiment_name
    elif 'prompttune' in experiment_name:
        experiment = prompt_tuning
        name += experiment_name + '_'+multivariate_projection+'_'+agg
    elif 'lora' in experiment_name:
        experiment = lora
        name += experiment_name
    elif 'linearprobe' in experiment_name:
        experiment = linearprobe
        name += experiment_name
    else:
        raise NotImplementedError


    # experiment_files = {
    #     'AtrialFibrillation': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/AtrialFibrillation/AtrialFibrillation",
    #                            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/AtrialFibrillation/AtrialFibrillation"),
    #     'SelfRegulationSCP1': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP1/SelfRegulationSCP1",
    #                            "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP1/SelfRegulationSCP1"),
    #     'SelfRegulationSCP2': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP2/SelfRegulationSCP2",
    #                 "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP2/SelfRegulationSCP2"),
    #     'Heartbeat': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Heartbeat/Heartbeat",
    #                 "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Heartbeat/Heartbeat"),
    #     'MotorImagery': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/MotorImagery/MotorImagery",
    #                 "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/MotorImagery/MotorImagery"),
    #              }

    # from gpt4ts
    experiment_files = {
        'JapaneseVowels': "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/JapaneseVowels/JapaneseVowels",
        'SelfRegulationSCP1': "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP1/SelfRegulationSCP1",
        'SelfRegulationSCP2': "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP2/SelfRegulationSCP2",
        'SpokenArabicDigits': "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SpokenArabicDigits/SpokenArabicDigits",
        'UWaveGestureLibrary': "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/UWaveGestureLibrary/UWaveGestureLibrary",
    }


    # check file exist
    assert all([os.path.exists(file+'_TEST.ts') for file in experiment_files.values()])

    random_seeds = [0, 1, 2, 3, 4] if bootstrap else [13]
    time = str(datetime.datetime.now())

    for dataset_name, filename in experiment_files.items():
        for seed in random_seeds:
            control_randomness(seed=seed)
            name_ = name + '/' + task + '/' + dataset_name

            train_loader, val_loader, test_loader = get_data(batch_size=batch_size, task=task, filename=filename)

            # try:
            _ = experiment(train_loader, val_loader, test_loader,
                           model_name, task,
                            name=name_, extra=time+'/'+str(seed),
                            epochs=epochs,
                            multivariate_projection=multivariate_projection,
                            agg=agg,
                            save_model=save_model, num_prefix=num_prefix,
                            flatten=flatten,
                            )
            # except Exception as e:
            #     print('error', e)
            #     continue





def long_forecast_experiments(experiment_name: str,
                              model_name: str,
                              multivariate_projection: str,
                              agg: str,
                              epochs: int = 10,
                              save_model: bool = False,
                              bootstrap: bool = False,
                              num_prefix: int = 16,
                              flatten: bool = False):

    from data import get_data
    from experiments import finetune, prompt_tuning, lora, linearprobe


    task = 'forecasting'
    batch_size = 1

    name = model_name + '/'
    if 'finetune_prompt' in experiment_name:
        experiment = prompt_tuning
        name += experiment_name
    elif 'finetune' in experiment_name:
        experiment = finetune
        name += experiment_name
    elif 'prompttune' in experiment_name:
        experiment = prompt_tuning
        name += experiment_name + '_'+multivariate_projection + '_' + agg
    elif 'lora' in experiment_name:
        experiment = lora
        name += experiment_name
    elif 'linearprobe' in experiment_name:
        experiment = linearprobe
        name += experiment_name
    else:
        raise NotImplementedError


    experiment_files = {
        # 'ETTm2': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv", [96]),
        # 'ETTm1': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTm1.csv", [96]),
        'national_illness': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/national_illness.csv",
                            #  [24, 60]),  # TODO: need forecast horizon 24 or 60
                             [60]),
        # 'ETTh1': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv", [96]),
        # 'ETTh2': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh2.csv", [96]),
        # 'exchange_rate': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/exchange_rate.csv", [96]),
    }
    # check file exist
    assert all([os.path.exists(file) for file, _ in experiment_files.values()])

    random_seeds = [0, 1, 2, 3, 4] if bootstrap else [13]
    time = str(datetime.datetime.now())
    print('time', time)


    for dataset_name, (filename, horizons) in experiment_files.items():
        for horizon in horizons:
            for seed in random_seeds:

                name_ = name + '/' + task + '/' + dataset_name + '/' + str(horizon)
                control_randomness(seed=seed)

                train_loader, val_loader, test_loader = get_data(batch_size=batch_size, task=task,
                                                                    filename=filename, forecast_horizon=horizon,)

                # try:
                _ = experiment(train_loader, val_loader, test_loader, model_name, task,
                                multivariate_projection=multivariate_projection,
                                agg=agg,
                                name=name_, extra=time+'/'+str(seed),
                                epochs=epochs,
                                save_model=save_model, forecast_horizon=horizon,
                                num_prefix=num_prefix, flatten=flatten,
                                )
                # except Exception as e:
                #     print('error', e)
                #     continue




def mimic_experiments(experiment_name: str,
                         model_name: str,
                         benchmark: str,
                         multivariate_projection: str,
                         agg: str,
                         epochs: int = 10,
                         save_model: bool = False,
                         bootstrap: bool = False,
                         num_prefix: int = 16,
                         flatten: bool = False):

    from data import load_mimic
    from experiments import finetune, prompt_tuning, lora, linearprobe

    batch_size = 1
    task = 'classification'
    equal_length = False
    small_part = True
    ordinal = True

    name = model_name + '/'
    if 'finetune_prompt' in experiment_name:
        experiment = prompt_tuning
        name += experiment_name
    elif 'finetune' in experiment_name:
        experiment = finetune
        name += experiment_name
    elif 'prompttune' in experiment_name:
        experiment = prompt_tuning
        name += experiment_name + '_'+multivariate_projection + '_' + agg
    elif 'lora' in experiment_name:
        experiment = lora
        name += experiment_name
    elif 'linearprobe' in experiment_name:
        experiment = linearprobe
        name += experiment_name
    else:
        raise NotImplementedError



    random_seeds = [0, 1, 2, 3, 4] if bootstrap else [13]
    time = str(datetime.datetime.now())

    name_ = name + '/mimic/' + benchmark
    for seed in random_seeds:
    
        control_randomness(seed=seed)

        train_loader, val_loader, test_loader = load_mimic(equal_length=equal_length, small_part=small_part,
                                                        benchmark=benchmark, ordinal=ordinal, seed=seed, batch_size=batch_size)

        # try:
        _ = experiment(train_loader, val_loader, test_loader, model_name, task,
                        name=name_, extra=time+'/'+str(seed),
                        epochs=epochs,
                        multivariate_projection=multivariate_projection,
                        agg=agg,
                        save_model=save_model, num_prefix=num_prefix,
                        flatten=flatten,
                        )
        
        
        # except Exception as e:
        #     print('error', e)
        #     continue






def run_baselines(experiment_name: str,
                    model_name: str,
                    benchmark: str,
                    multivariate_projection: str,
                    agg: str,
                    epochs: int = 10,
                    save_model: bool = False,
                    bootstrap: bool = False,
                    num_prefix: int = 16,
                    flatten: bool = False):
    # Classification:
        # Logistic regression, MLP
        # TimesNet (non-transformer, DL)
    # Forecasting:
        # PatchTST (transformer)
        # TimesNet
        # linear regression

    model_name = 'lr'




    pass







if __name__ == "__main__":

    from data import get_data
    from experiments import finetune, prompt_tuning



    # torch.autograd.set_detect_anomaly(True)

    # create dir if not exist
    os.makedirs('/home/scratch/mingzhul/moment-research/results/wandb/', exist_ok=True)

    # suffix = ''
    suffix = 'focal_loss_'
    # suffix = 'cost_sensitive_loss'
    save_model = False
    bootstrap = True
    epochs = 10

    # os.environ["WANDB_MODE"] = "offline"

    model_name = 'moment'
    # model_name = 'gpt4ts'


    flatten = False
    categorical_embedding = False

    EXPERIMENT_NAME = 'prompttune'
    # EXPERIMENT_NAME = 'finetune'
    # EXPERIMENT_NAME = 'lora'
    # EXPERIMENT_NAME = 'linearprobe'

    # EXPERIMENT_NAME = 'finetune_prompt'

    EXPERIMENT_NAME = suffix + EXPERIMENT_NAME

    multivariate_projection = 'attention'
    # multivariate_projection = 'vanilla'  # p-tuning v2
    # multivariate_projection = 'vanilla_vanilla' # original prompt tuning

    agg = 'mlp'
    # agg = 'rnn'

    # NOTE: forecasting use 16, classification use 4
    num_prefix = 4
    # num_prefix = 32


    # classify_experiments(EXPERIMENT_NAME, model_name, multivariate_projection=multivariate_projection, agg=agg,
    #                      epochs=epochs, save_model=save_model, bootstrap=bootstrap, num_prefix=num_prefix, flatten=flatten)  # this one probably needs larger gpu
    # long_forecast_experiments(EXPERIMENT_NAME, model_name, multivariate_projection=multivariate_projection, agg=agg,
    #                           epochs=epochs, save_model=save_model, bootstrap=bootstrap, num_prefix=num_prefix, flatten=flatten)


    benchmark = 'mortality'
    # benchmark = 'phenotyping'
    mimic_experiments(EXPERIMENT_NAME, model_name, benchmark, multivariate_projection=multivariate_projection, agg=agg,
                      epochs=epochs, save_model=save_model, bootstrap=bootstrap, num_prefix=num_prefix, flatten=flatten)





    # TODO: try inputing ones to the prompt module, should recover vanilla prompt tuning?
    # TODO: class balance should be same for train val test


    # TODO: balance weights for classification

    # TODO: should you do revin for categorical data?
    #   revin does std, so categorical numbers are very big after std, if one batch only has one category


    # TODO:

    # instead of mimic, what about just sample?

    # finetune + multivariable?
    #  - what's the point? just use new model that can handle multiple variables

    # TODO: look at notes for details
    # Use date as covariate??? so you can have multiple variables for univariate time series?? to deal with non-constant frequency
    # TODO: does this really help in RNNs?

    # why is fintuning perform similarly to moment paper results?

    # do prompt tuning on other models
    # survival analysis

    # TODO: compare your multivariable code to itransformer


    # TODO ablation:
    # adding increasing number of channels



