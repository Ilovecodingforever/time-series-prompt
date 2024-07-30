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


import os
import torch

from momentfm.utils.utils import control_randomness

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 0


# Set random seeds for PyTorch, Numpy etc.
control_randomness(seed=RANDOM_SEED)










def classify_experiments(experiment_name, multivariate_projection='attention', epochs=20):
    from data import get_data
    from experiments import zero_shot, finetune, prompt_tuning


    multitask = False
    multivariable = True
    no_train_forehead = False

    dataset_names = ['classify']


    batch_size = 1

    name = ''
    if experiment_name == 'zero_shot':
        experiment = zero_shot
    elif experiment_name == 'finetune':
        experiment = finetune
        name = 'finetune'
    elif experiment_name == 'prompt_tuning':
        experiment = prompt_tuning
        name = 'prompttune_'+multivariate_projection
    else:
        raise NotImplementedError


    experiment_files = {
        'AtrialFibrillation': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/AtrialFibrillation/AtrialFibrillation_TEST.ts",
                               "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/AtrialFibrillation/AtrialFibrillation_TRAIN.ts"),
        'Epilepsy': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Epilepsy/Epilepsy_TEST.ts",
                    "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Epilepsy/Epilepsy_TRAIN.ts"),
        'ERing': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/ERing/ERing_TEST.ts",
                    "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/ERing/ERing_TRAIN.ts"),
        'Cricket': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Cricket/Cricket_TEST.ts",
                    "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Cricket/Cricket_TRAIN.ts"),
        'SelfRegulationSCP1': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP1/SelfRegulationSCP1_TEST.ts",
                               "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP1/SelfRegulationSCP1_TRAIN.ts"),
        'SelfRegulationSCP2': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP2/SelfRegulationSCP2_TEST.ts",
                    "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.ts"),
        'Heartbeat': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Heartbeat/Heartbeat_TEST.ts",
                    "/zfsauton/project/public/Mononito/TimeseriesDatasets/classification/UCR/Heartbeat/Heartbeat_TRAIN.ts"),
                 }

    for dataset_name, files in experiment_files.items():
        # check file exist
        assert all([os.path.exists(file) for file in files])


    for dataset_name, files in experiment_files.items():
        name_ = name + '/' + dataset_names[0] + '/' + dataset_name

        train_loader, test_loader = get_data(batch_size=batch_size, dataset_names=dataset_names, all=True,
                                            files=files)

        model = experiment(train_loader, test_loader, name_,
                        prefix_tuning_multi=multivariable,
                        MPT=multitask,
                        no_train_forehead=no_train_forehead,
                        epochs=epochs,
                        multivariate_projection=multivariate_projection
                        )





def informer_experiments(dataset_names, experiment_name, multivariate_projection='attention', epochs=20,):

    from data import get_data
    from experiments import zero_shot, finetune, prompt_tuning


    multitask = False
    multivariable = True
    no_train_forehead = False

    batch_size = 1

    name = ''
    if experiment_name == 'zero_shot':
        experiment = zero_shot
    elif experiment_name == 'finetune':
        experiment = finetune
        name = 'finetune'
    elif experiment_name == 'prompt_tuning':
        experiment = prompt_tuning
        name = 'prompttune_'+multivariate_projection
    else:
        raise NotImplementedError


    experiment_files = {
        'ETTm1': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTm1.csv", ),
        'ETTm2': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv", ),
        'ETTh1': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv", ),
        'ETTh2': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh2.csv", ),
        'exchange_rate': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/exchange_rate.csv", ),
        # 'national_illness': ("/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/national_illness.csv", ),  # TODO: need forecast horizon 24 or 60
    }

    for dataset_name, files in experiment_files.items():
        # check file exist
        assert all([os.path.exists(file) for file in files])


    for dataset_name, files in experiment_files.items():
        name_ = name + '/' + dataset_names[0] + '/' + dataset_name

        train_loader, test_loader = get_data(batch_size=batch_size, dataset_names=dataset_names, all=True,
                                            files=files)

        model = experiment(train_loader, test_loader, name_,
                        prefix_tuning_multi=multivariable,
                        MPT=multitask,
                        no_train_forehead=no_train_forehead,
                        epochs=epochs,
                        multivariate_projection=multivariate_projection
                        )


def long_forecast_experiments(experiment_name, multivariate_projection='attention', epochs=20):

    dataset_names = ['forecasting_long']

    informer_experiments(dataset_names, experiment_name, multivariate_projection=multivariate_projection, epochs=epochs)
                    



def imputation_experiments(experiment_name, multivariate_projection='attention', epochs=20):

    dataset_names = ['imputation']

    informer_experiments(dataset_names, experiment_name, multivariate_projection=multivariate_projection, epochs=epochs)





def multitask_experiments(experiment_name, epochs=20):
    from data import get_data
    from experiments import zero_shot, finetune, prompt_tuning


    multitask = True
    multivariable = False
    no_train_forehead = False

    dataset_names = ['forecasting_short', 'classify', 'anomaly', 'forecasting_long', 'imputation', ]


    batch_size = 1

    name = ''
    if experiment_name == 'zero_shot':
        experiment = zero_shot
    elif experiment_name == 'finetune':
        experiment = finetune
        name = 'finetune'
    elif experiment_name == 'prompt_tuning':
        experiment = prompt_tuning
        name = 'prompttune'
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



    for dataset_name, files in experiment_files.items():
        # check file exist
        assert all([os.path.exists(file) for file in files])


    train_loader, test_loader = get_data(batch_size=batch_size, dataset_names=dataset_names, all=True,
                                        files=experiment_files)

    model = experiment(train_loader, test_loader, 'MPT_'+name,
                    prefix_tuning_multi=multivariable,
                    MPT=multitask,
                    no_train_forehead=no_train_forehead,
                    epochs=epochs,
                    )



if __name__ == "__main__":
    from data import get_data
    from experiments import zero_shot, finetune, prompt_tuning

    # torch.autograd.set_detect_anomaly(True)

    # os.environ["WANDB_MODE"] = "offline"

    # EXPERIMENT_NAME = 'zero_shot'
    EXPERIMENT_NAME = 'prompt_tuning'
    # EXPERIMENT_NAME = 'finetune'

    multivariate_projection = 'linear'
    # multivariate_projection = 'attention'
    

    # classify_experiments(EXPERIMENT_NAME, multivariate_projection=multivariate_projection, epochs=20)  # this one probably needs larger gpu
    # imputation_experiments(EXPERIMENT_NAME, multivariate_projection=multivariate_projection, epochs=20)
    # long_forecast_experiments(EXPERIMENT_NAME, multivariate_projection=multivariate_projection, epochs=20)
    multitask_experiments(EXPERIMENT_NAME, epochs=20)

    # TODO: short horizon loss should be smape
    # TODO: all data similar size
    # TODO: average all performance for anomaly



    

    # all = True

    # multitask = False
    # multivariable = True

    # no_train_forehead = False

    # dataset_names = ['imputation', 'anomaly', 'classify', 'forecasting_long']

    # if not multitask and not multivariable and EXPERIMENT_NAME == 'prompt_tuning':
    #     dataset_names = ['forecasting_long']

    # batch_size = 1

    # name = ''
    # if EXPERIMENT_NAME == 'zero_shot':
    #     experiment = zero_shot
    # elif EXPERIMENT_NAME == 'finetune':
    #     experiment = finetune
    # elif EXPERIMENT_NAME == 'prompt_tuning':
    #     experiment = prompt_tuning
    #     name = f'_multitask_{multitask}_multivariable_{multivariable}'
    # else:
    #     raise NotImplementedError

