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

from momentfm.utils.utils import control_randomness

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 13


# Set random seeds for PyTorch, Numpy etc.
control_randomness(seed=RANDOM_SEED)










def classify_experiments(experiment_name, multivariate_projection='attention', epochs=20,
                         save_model=False, bootstrap=False):
    from data import get_data
    from experiments import zero_shot, finetune, prompt_tuning, lora, linearprobe


    multitask = False
    multivariable = True
    no_train_forehead = False

    dataset_names = ['classify']


    batch_size = 1

    name = ''
    if 'zero_shot' in experiment_name:
        experiment = zero_shot
    elif 'finetune' in experiment_name:
        experiment = finetune
        name = experiment_name
    elif 'prompttune' in experiment_name:
        experiment = prompt_tuning
        name = experiment_name + '_'+multivariate_projection
    elif 'lora' in experiment_name:
        experiment = lora
        name = experiment_name
    else:
        raise NotImplementedError


    experiment_files = {
        'AtrialFibrillation': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/AtrialFibrillation/AtrialFibrillation_TEST.ts",
                               "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/AtrialFibrillation/AtrialFibrillation_TRAIN.ts"),
        # 'Epilepsy': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Epilepsy/Epilepsy_TEST.ts",
        #             "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Epilepsy/Epilepsy_TRAIN.ts"),
        # 'ERing': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/ERing/ERing_TEST.ts",
        #             "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/ERing/ERing_TRAIN.ts"),
        # 'Cricket': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Cricket/Cricket_TEST.ts",
        #             "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Cricket/Cricket_TRAIN.ts"),
        'SelfRegulationSCP1': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP1/SelfRegulationSCP1_TEST.ts",
                               "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP1/SelfRegulationSCP1_TRAIN.ts"),
        'SelfRegulationSCP2': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP2/SelfRegulationSCP2_TEST.ts",
                    "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.ts"),
        'Heartbeat': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Heartbeat/Heartbeat_TEST.ts",
                    "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Heartbeat/Heartbeat_TRAIN.ts"),
        'MotorImagery': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/MotorImagery/MotorImagery_TEST.ts",
                    "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/MotorImagery/MotorImagery_TRAIN.ts"),
                 }

    for dataset_name, files in experiment_files.items():
        # check file exist
        assert all([os.path.exists(file) for file in files])

    random_seeds = [0, 1, 2, 3, 4] if bootstrap else [13]
    time = str(datetime.datetime.now())

    for seed in random_seeds:
        control_randomness(seed=seed)
        for dataset_name, files in experiment_files.items():
            name_ = name + '/' + dataset_names[0] + '/' + dataset_name

            train_loader, val_loader, test_loader = get_data(batch_size=batch_size, dataset_names=dataset_names, all=True,
                                                                files=files)

            model = experiment(train_loader, val_loader, test_loader, name_, time+'/'+str(seed),
                                prefix_tuning_multi=multivariable,
                                MPT=multitask,
                                no_train_forehead=no_train_forehead,
                                epochs=epochs,
                                multivariate_projection=multivariate_projection,
                                save_model=save_model
                                )





def informer_experiments(dataset_names, experiment_name, multivariate_projection='attention', epochs=20,
                         save_model=False, bootstrap=False, num_prefix=16):

    from data import get_data
    from experiments import zero_shot, finetune, prompt_tuning, lora, linearprobe


    multitask = False
    multivariable = True
    no_train_forehead = False

    batch_size = 1

    name = ''
    if 'zero_shot' in experiment_name:
        experiment = zero_shot
    elif 'finetune_prompt' in experiment_name:
        experiment = prompt_tuning
        name = experiment_name
    elif 'finetune' in experiment_name:
        experiment = finetune
        name = experiment_name
    elif 'prompttune' in experiment_name:
        experiment = prompt_tuning
        name = experiment_name + '_'+multivariate_projection
    elif 'lora' in experiment_name:
        experiment = lora
        name = experiment_name
    elif 'linearprobe' in experiment_name:
        experiment = linearprobe
        name = experiment_name
    else:
        raise NotImplementedError


    experiment_files = {
        # 'ETTm2': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv", ),
        # 'ETTm1': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTm1.csv", ),
        # 'ETTh1': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv", ),
        # 'ETTh2': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh2.csv", ),
        # 'exchange_rate': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/exchange_rate.csv", ),
        'national_illness': (("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/national_illness.csv", ),
                            #  [24, 60]),  # TODO: need forecast horizon 24 or 60
                             [60])
    }

    for dataset_name, (files, horizons) in experiment_files.items():
        # check file exist
        assert all([os.path.exists(file) for file in files])


    random_seeds = [0, 1, 2, 3, 4] if bootstrap else [13]
    time = str(datetime.datetime.now())
    print('time', time)


    mask_ratios = [0.125, 0.25, 0.3, 0.375, 0.5]
    if dataset_names[0] != 'imputation':
        mask_ratios = [0.3]



    for seed in random_seeds:
        control_randomness(seed=seed)

        for dataset_name, (files, horizons) in experiment_files.items():

            if dataset_names[0] != 'forecasting_long':
                # no need to run more than once
                horizons = [horizons[0]]

            for mask_ratio in mask_ratios:

                for horizon in horizons:
                    name_ = name + '/' + dataset_names[0] + '/' + dataset_name
                    if dataset_names[0] == 'forecasting_long':
                        name_ += '/' + str(horizon)
                    if dataset_names[0] == 'imputation':
                        name_ += '/' + str(mask_ratio)

                    train_loader, val_loader, test_loader = get_data(batch_size=batch_size, dataset_names=dataset_names, all=True,
                                                                    files=files, forecast_horizon=horizon)
                    # TODO: why do I need to reset the dataset? isn't it already new?
                    train_loader.dataset.reset()
                    val_loader.dataset.reset()
                    test_loader.dataset.reset()

                    model = experiment(train_loader, val_loader, test_loader,
                                       name=name_, extra=time+'/'+str(seed),
                                        prefix_tuning_multi=multivariable,
                                        MPT=multitask,
                                        no_train_forehead=no_train_forehead,
                                        epochs=epochs,
                                        multivariate_projection=multivariate_projection,
                                        save_model=save_model, forecast_horizon=horizon,
                                        mask_ratio=mask_ratio, num_prefix=num_prefix,
                                        )


def long_forecast_experiments(experiment_name, multivariate_projection='attention', epochs=20, save_model=False, bootstrap=False, num_prefix=16):

    dataset_names = ['forecasting_long']

    informer_experiments(dataset_names, experiment_name, multivariate_projection=multivariate_projection, epochs=epochs,
                         save_model=save_model, bootstrap=bootstrap, num_prefix=num_prefix)




def imputation_experiments(experiment_name, multivariate_projection='attention', epochs=20, save_model=False, bootstrap=False):

    dataset_names = ['imputation']

    informer_experiments(dataset_names, experiment_name, multivariate_projection=multivariate_projection, epochs=epochs,
                         save_model=save_model, bootstrap=bootstrap)





def multitask_experiments(experiment_name, epochs=20, save_model=False):
    from data import get_data
    from experiments import zero_shot, finetune, prompt_tuning, lora


    multitask = True
    multivariable = False
    no_train_forehead = False



    batch_size = 16

    name = ''
    if 'zero_shot' in experiment_name:
        experiment = zero_shot
    elif 'finetune' in experiment_name:
        experiment = finetune
        name = experiment_name
    elif 'prompttune' in experiment_name:
        experiment = prompt_tuning
        name = experiment_name
    elif 'lora' in experiment_name:
        experiment = lora
        name = experiment_name
    else:
        raise NotImplementedError


    experiment_files = {}

    # TODO: use fred data? /zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/fred
    experiment_files['forecasting_short'] = (
                                        '/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/monash/m4_yearly_dataset.tsf',
                                        #   TODO: too much data already, also forecast horizon don't match
                                        #   '/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/monash/m4_quarterly_dataset.tsf',
                                        #   '/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/monash/m4_monthly_dataset.tsf',
                                        #   '/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/monash/m3_yearly_dataset.tsf',
                                        #   '/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/monash/m3_quarterly_dataset.tsf',
                                        #   '/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/monash/m3_monthly_dataset.tsf',
                                          )

    experiment_files['imputation'] = ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv",
                                      "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh2.csv",
                                      "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTm1.csv",
                                      "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv",
                                      "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/exchange_rate.csv", )

    experiment_files['forecasting_long'] = ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv",
                                            "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh2.csv",
                                            "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTm1.csv",
                                            "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv",
                                            "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/exchange_rate.csv", )

    experiment_files['classify'] = (
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/AtrialFibrillation/AtrialFibrillation_TEST.ts",
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/AtrialFibrillation/AtrialFibrillation_TRAIN.ts",
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Epilepsy/Epilepsy_TEST.ts",
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Epilepsy/Epilepsy_TRAIN.ts",
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/ERing/ERing_TEST.ts",
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/ERing/ERing_TRAIN.ts",
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Cricket/Cricket_TEST.ts",
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Cricket/Cricket_TRAIN.ts",
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP1/SelfRegulationSCP1_TEST.ts",
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP1/SelfRegulationSCP1_TRAIN.ts",
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP2/SelfRegulationSCP2_TEST.ts",
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.ts",
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Heartbeat/Heartbeat_TEST.ts",
        "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Heartbeat/Heartbeat_TRAIN.ts")


    experiment_files['anomaly'] = (
                                   '/zfsauton/project/public/Mononito/TimeseriesDatasets/anomaly_detection/TSB-UAD-Public/MITDB',
                                #    '/zfsauton/project/public/Mononito/TimeseriesDatasets/anomaly_detection/TSB-UAD-Public/ECG'  # TOO big
                                # '/zfsauton/project/public/Mononito/TimeseriesDatasets/anomaly_detection/TSB-UAD-Public/SVDB',
                                   )

    dataset_names = experiment_files.keys()

    for dataset_name, files in experiment_files.items():
        # check file exist
        assert all([os.path.exists(file) for file in files])


    train_loader, val_loader, test_loader = get_data(batch_size=batch_size, dataset_names=dataset_names, all=True,
                                                        files=experiment_files)

    model = experiment(train_loader, val_loader, test_loader, 'MPT_'+name,
                    prefix_tuning_multi=multivariable,
                    MPT=multitask,
                    no_train_forehead=no_train_forehead,
                    epochs=epochs,
                    save_model=save_model
                    )



if __name__ == "__main__":

    from data import get_data
    from experiments import zero_shot, finetune, prompt_tuning

    # torch.autograd.set_detect_anomaly(True)

    suffix = ''
    save_model = False

    # os.environ["WANDB_MODE"] = "offline"

    # EXPERIMENT_NAME = 'zero_shot'
    EXPERIMENT_NAME = 'prompttune'
    # EXPERIMENT_NAME = 'finetune'
    # EXPERIMENT_NAME = 'lora'
    # EXPERIMENT_NAME = 'linearprobe'
    # EXPERIMENT_NAME = 'finetune_prompt'


    EXPERIMENT_NAME = suffix + EXPERIMENT_NAME

    # multivariate_projection = 'linear'
    multivariate_projection = 'attention'
    # multivariate_projection = 'vanilla'
    # multivariate_projection = 'residual'

    num_prefix = 8

    bootstrap = True

    epochs = 10

    # classify_experiments(EXPERIMENT_NAME, multivariate_projection=multivariate_projection, epochs=epochs,
    #                         save_model=save_model, bootstrap=bootstrap)  # this one probably needs larger gpu
    # imputation_experiments(EXPERIMENT_NAME, multivariate_projection=multivariate_projection, epochs=epochs,
    #                       save_model=save_model, bootstrap=bootstrap)
    long_forecast_experiments(EXPERIMENT_NAME, multivariate_projection=multivariate_projection, epochs=epochs,
                              save_model=save_model, bootstrap=bootstrap, num_prefix=num_prefix)
    # multitask_experiments(EXPERIMENT_NAME, epochs=epochs, save_model=save_model)



    # TODO:

    # instead of mimic, what about just sample?

    # finetune + multivariable?
    #  - what's the point? just use new model that can handle multiple variables


    # TODO: look at notes for details
    # Use date as covariate??? so you can have multiple variables for univariate time series?? to deal with non-constant frequency
    # TODO: does this really help in RNNs?

    # why is fintuning perform similarly to moment paper results?

    # TODO: anomaly vus AUC

    # TODO: visualize prompt

    # TODO: debug linear multivariable

    # do prompt tuning on other models
    # survival analysis

    # TODO: do parallel processing, research code is a lot faster. maybe because of autocast? changed current code to match research code but haven't run yet

    # TODO: research code end-to-end finetune is with t5-base?

    # TODO: allow different batch size for different tasks

    # TODO: all data similar size

    # TODO: compare your multivariable code to itransformer


    # TODO ablation:
    # adding increasing number of channels


    # done:
    # what should stride be?
    # they are all 1 in moment research code
