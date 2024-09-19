"""
load performance from files
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime



# TODO: 10 or 20? for epochs
epochs = 20



def print_multivariable_performance(dir, load_test=True):

    if load_test:
        test_performance = {}
        for task in ['classify', 'imputation', 'forecasting_long']:
        # for task in ['forecasting_long']:
            test_performance[task] = {}
            for dataset in sorted(os.listdir(f"{dir}/{task}")):
                # select latest
                times = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in os.listdir(f"{dir}/{task}/{dataset}")]
                time = max(times)
                # to string
                time = time.strftime("%Y-%m-%d %H:%M:%S.%f")

                d = f"{dir}/{task}/{dataset}/{time}/test/0/{task}"

                files = sorted(os.listdir(d))
                for file in files:
                    with open(f"{d}/{file}", "rb") as f:
                        performance = pickle.load(f)

                        if task == 'classify':
                            file = file.split("_")[0]

                        if file not in test_performance[task]:
                            test_performance[task][file] = []
                        test_performance[task][file].append(performance[task])


        print("test")
        for task in test_performance:
            for file in test_performance[task]:
                print(f"{task} {file}")
                for metric in test_performance[task][file][0]:
                    print(metric, test_performance[task][file][0][metric])

        return

    train_performance = {}
    test_performance = {}

    # write to file
    with open(f"{dir}/performance_{epochs}.txt", "w") as p_f:

        for task in ['classify', 'imputation', 'forecasting_long']:

            # TODO
            # if task == 'forecasting_long':
            #     continue

            for dataset in sorted(os.listdir(f"{dir}/{task}")):
                # select latest
                times = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in os.listdir(f"{dir}/{task}/{dataset}")]
                time = max(times)
                # to string
                time = time.strftime("%Y-%m-%d %H:%M:%S.%f")

                # train
                train_performance[task] = {}
                for epoch in range(epochs):
                    d = f"{dir}/{task}/{dataset}/{time}/train/{epoch}/{task}"
                    files = sorted(os.listdir(d))
                    for file in files:
                        with open(f"{d}/{file}", "rb") as f:
                            performance = pickle.load(f)

                            if task == 'classify':
                                file = file.split("_")[0]

                            if file not in train_performance[task]:
                                train_performance[task][file] = []
                            train_performance[task][file].append(performance[task])

                assert len(train_performance[task][file]) == epochs

                # convert to {key: []}
                train_performance[task][file] = {k: [train_performance[task][file][i][k] for i in range(epochs)] for k in train_performance[task][file][0]}
                train_performance[task][file]['best epoch'] = np.where(np.array(train_performance[task][file]['loss']) == min(train_performance[task][file]['loss']))[0][0]

                print(f"{task} {dataset} {file} \ntrain best epoch {train_performance[task][file]['best epoch']}", file=p_f)
                for metric in train_performance[task][file]:
                    if metric != 'best epoch':
                        print(metric, train_performance[task][file][metric][train_performance[task][file]['best epoch']], file=p_f)

                # test
                test_performance[task] = {}
                for epoch in range(epochs):
                    d = f"{dir}/{task}/{dataset}/{time}/test/{epoch}/{task}"
                    files = sorted(os.listdir(d))
                    for file in files:
                        with open(f"{d}/{file}", "rb") as f:
                            performance = pickle.load(f)

                            if task == 'classify':
                                file = file.split("_")[0]

                            if file not in test_performance[task]:
                                test_performance[task][file] = []
                            test_performance[task][file].append(performance[task])

                # convert to {key: []}
                test_performance[task][file] = {k: [test_performance[task][file][i][k] for i in range(epochs)] for k in test_performance[task][file][0]}


                print("test", file=p_f)
                for metric in test_performance[task][file]:
                    print(metric, test_performance[task][file][metric][train_performance[task][file]['best epoch']], file=p_f)



    print()





def print_mpt_performance(dir, time=None, load_test=False):
    if time is None:
        # select latest
        times = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in os.listdir(dir)]
        time = max(times)
        # to string
        time = time.strftime("%Y-%m-%d %H:%M:%S.%f")



    if load_test:



        return




    # write to file
    with open(f"{dir}/performance_{epochs}.txt", "w") as p_f:

        train_performance = {}
        test_performance = {}
        tasks = ['anomaly', 'classify', 'forecasting_long', 'forecasting_short', 'imputation']

        # train
        for task in tasks:
            train_performance[task] = {}

            for epoch in range(epochs):
                d = f"{dir}/{time}/train/{epoch}"
                files = sorted(os.listdir(f"{d}/{task}"))
                for file in files:
                    with open(f"{d}/{task}/{file}", "rb") as f:

                        if task == 'classify':
                            file = file.split("_")[0]


                        performance = pickle.load(f)
                        if file not in train_performance[task]:
                            train_performance[task][file] = []
                        train_performance[task][file].append(performance[task])

        # convert to {key: []}
            for file in train_performance[task]:
                train_performance[task][file] = {k: [train_performance[task][file][i][k] for i in range(epochs)] for k in train_performance[task][file][0]}
                train_performance[task][file]['best epoch'] = np.where(np.array(train_performance[task][file]['loss']) == min(train_performance[task][file]['loss']))[0][0]

        train_performance['anomaly']['average MITDB'] = {}
        metrics = ['adj F1', 'loss']
        for metric in metrics:
            train_performance['anomaly']['average MITDB'][metric] = []
            for epoch in range(epochs):
                lst = []
                for file in train_performance['anomaly']:
                    if file == 'average MITDB':
                        continue
                    # TODO: ignore 0?
                    if metric == 'adj F1' and train_performance['anomaly'][file][metric][epoch] == 0:
                        continue
                    lst.append(train_performance['anomaly'][file][metric][epoch])

                train_performance['anomaly']['average MITDB'][metric].append(np.mean(lst))

        train_performance['anomaly']['average MITDB']['best epoch'] = np.where(np.array(train_performance['anomaly']['average MITDB']['loss']) == min(train_performance['anomaly']['average MITDB']['loss']))[0][0]

        # test
        for task in tasks:
            test_performance[task] = {}

            for epoch in range(epochs):
                d = f"{dir}/{time}/test/{epoch}"
                files = sorted(os.listdir(f"{d}/{task}"))
                for file in files:
                    with open(f"{d}/{task}/{file}", "rb") as f:

                        if task == 'classify':
                            file = file.split("_")[0]

                        performance = pickle.load(f)
                        if file not in test_performance[task]:
                            test_performance[task][file] = []
                        test_performance[task][file].append(performance[task])

        # convert to {key: []}
        for task in tasks:
            for file in test_performance[task]:
                test_performance[task][file] = {k: [test_performance[task][file][i][k] for i in range(epochs)] for k in test_performance[task][file][0]}

        test_performance['anomaly']['average MITDB'] = {}
        metrics = ['adj F1', 'loss']
        for metric in metrics:
            test_performance['anomaly']['average MITDB'][metric] = []
            for epoch in range(epochs):
                lst = []
                for file in test_performance['anomaly']:
                    if file == 'average MITDB':
                        continue
                    # TODO: ignore 0?
                    if metric == 'adj F1' and test_performance['anomaly'][file][metric][epoch] == 0:
                        continue
                    lst.append(test_performance['anomaly'][file][metric][epoch])

                test_performance['anomaly']['average MITDB'][metric].append(np.mean(lst))


        for task in tasks:
            for file in test_performance[task]:
                print(f"{task} {file} \ntrain best epoch {train_performance[task][file]['best epoch']}", file=p_f)
                for metric in train_performance[task][file]:
                    if metric != 'best epoch':
                        print(metric, train_performance[task][file][metric][train_performance[task][file]['best epoch']], file=p_f)

                print("test", file=p_f)
                for metric in test_performance[task][file]:
                    print(metric, test_performance[task][file][metric][train_performance[task][file]['best epoch']], file=p_f)

    print()








def print_stuff(files):
    for f in files:
        # read pickle
        print(f)
        with open(f, "rb") as f:
            performance = pickle.load(f)
            print(performance)





def bootstrap():
    dirs = [
        ('forecasting_long', 'performance/finetune/forecasting_long/ETTh1/96/2024-09-14 12:29:12.019549', 'autoformer_ETTh1.csv.pkl'),
        ('forecasting_long', 'performance/finetune/forecasting_long/ETTh2/96/2024-09-14 12:29:12.019549', 'autoformer_ETTh2.csv.pkl'),
        ('forecasting_long', 'performance/finetune/forecasting_long/exchange_rate/96/2024-09-14 12:29:12.019549', 'autoformer_exchange_rate.csv.pkl'),
        ('forecasting_long', 'performance/finetune/forecasting_long/national_illness/60/2024-09-14 12:29:12.019549', 'autoformer_national_illness.csv.pkl'),
    ]


    for task, dir, filename in dirs:
        performances = []
        for seed in os.listdir(dir):
            try:
                with open(f"{dir}/{seed}/test/0/{task}/{filename}", "rb") as f:
                    performance = pickle.load(f)
                    # print(performance)
                performances.append(performance[task])
            except Exception as e:
                print(e)
                
        # calculate mean and std
        print(dir)
        mean = {}
        std = {}
        if len(performances) == 0:
            mean = None
            std = None
        else:
            for metric in performances[0]:
                mean[metric] = np.mean([p[metric] for p in performances])
                std[metric] = np.std([p[metric] for p in performances])

        print(mean)
        print(std)


    pass
















if __name__ == "__main__":

    bootstrap()

    # files = [
    #     'performance/linearprobe/forecasting_long/national_illness/24/2024-08-15 22:24:19.583583/test/0/forecasting_long/autoformer_national_illness.csv.pkl',
    #     'performance/finetune/forecasting_long/national_illness/24/2024-08-15 22:23:25.783949/test/0/forecasting_long/autoformer_national_illness.csv.pkl',
    #     'performance/lora/forecasting_long/national_illness/24/2024-08-15 22:23:35.614756/test/0/forecasting_long/autoformer_national_illness.csv.pkl',

    #     'performance/linearprobe/forecasting_long/national_illness/60/2024-08-15 22:29:16.904540/test/0/forecasting_long/autoformer_national_illness.csv.pkl',
    #     'performance/finetune/forecasting_long/national_illness/60/2024-08-15 22:26:45.978880/test/0/forecasting_long/autoformer_national_illness.csv.pkl',
    #     'performance/lora/forecasting_long/national_illness/60/2024-08-15 22:26:05.767002/test/0/forecasting_long/autoformer_national_illness.csv.pkl',

    #     'performance/linearprobe/imputation/national_illness/2024-08-15 21:38:16.737470/test/0/imputation/autoformer_national_illness.csv.pkl',
    #     'performance/finetune/imputation/national_illness/2024-08-15 21:37:22.016527/test/0/imputation/autoformer_national_illness.csv.pkl',
    #     'performance/lora/imputation/national_illness/2024-08-15 21:37:43.314456/test/0/imputation/autoformer_national_illness.csv.pkl',

    #     'performance/lora/classify/AtrialFibrillation/2024-08-15 21:35:06.623782/test/0/classify/AtrialFibrillation_AtrialFibrillation_TEST.ts.pkl',
    #     'performance/lora/classify/Heartbeat/2024-08-15 21:51:01.401104/test/0/classify/Heartbeat_Heartbeat_TEST.ts.pkl',
    #     'performance/lora/classify/MotorImagery/2024-08-15 22:06:51.838860/test/0/classify/MotorImagery_MotorImagery_TEST.ts.pkl',
    #     'performance/lora/classify/SelfRegulationSCP1/2024-08-15 21:35:58.812546/test/0/classify/SelfRegulationSCP1_SelfRegulationSCP1_TEST.ts.pkl',
    #     'performance/lora/classify/SelfRegulationSCP2/2024-08-15 21:44:26.206698/test/0/classify/SelfRegulationSCP2_SelfRegulationSCP2_TEST.ts.pkl',

    #     'performance/finetune/classify/AtrialFibrillation/2024-08-15 21:31:09.616148/test/0/classify/AtrialFibrillation_AtrialFibrillation_TEST.ts.pkl',
    #     'performance/finetune/classify/Heartbeat/2024-08-15 21:49:34.783605/test/0/classify/Heartbeat_Heartbeat_TEST.ts.pkl',
    #     'performance/finetune/classify/MotorImagery/2024-08-15 22:07:08.529092/test/0/classify/MotorImagery_MotorImagery_TEST.ts.pkl',
    #     'performance/finetune/classify/SelfRegulationSCP1/2024-08-15 21:32:40.826703/test/0/classify/SelfRegulationSCP1_SelfRegulationSCP1_TEST.ts.pkl',
    #     'performance/finetune/classify/SelfRegulationSCP2/2024-08-15 21:42:00.377426/test/0/classify/SelfRegulationSCP2_SelfRegulationSCP2_TEST.ts.pkl',

    #     'performance/linearprobe/classify/AtrialFibrillation/2024-08-15 21:30:58.081670/test/0/classify/AtrialFibrillation_AtrialFibrillation_TEST.ts.pkl',
    #     'performance/linearprobe/classify/Heartbeat/2024-08-15 21:56:31.430663/test/0/classify/Heartbeat_Heartbeat_TEST.ts.pkl',
    #     'performance/linearprobe/classify/MotorImagery/2024-08-15 22:28:15.358816/test/0/classify/MotorImagery_MotorImagery_TEST.ts.pkl',
    #     'performance/linearprobe/classify/SelfRegulationSCP1/2024-08-15 21:33:13.330513/test/0/classify/SelfRegulationSCP1_SelfRegulationSCP1_TEST.ts.pkl',
    #     'performance/linearprobe/classify/SelfRegulationSCP2/2024-08-15 21:46:10.648634/test/0/classify/SelfRegulationSCP2_SelfRegulationSCP2_TEST.ts.pkl'
    # ]

    # print_stuff(files)



    # print_mpt_performance("performance/MPT_prompttune", time='2024-07-31 10:59:58.071013')
    # print_mpt_performance("performance/MPT_finetune", time='2024-07-31 10:59:58.100640')

    # print_multivariable_performance("performance/linearprobe")
    # print_multivariable_performance("performance/finetune")
    # print_multivariable_performance("performance/lora")
    # print_multivariable_performance("performance/prompttune_linear")



