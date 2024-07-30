"""
load performance from files
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime




def print_multivariable_performance(dir):
    train_performance = {}
    test_performance = {}
    
    # write to file
    with open(f"{dir}/performance.txt", "w") as f:
        
        for task in ['classify', 'imputation', 'forecasting_long']:

            # TODO
            if task == 'forecasting_long':
                continue

            for dataset in os.listdir(f"{dir}/{task}"):
                # select latest
                times = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in os.listdir(f"{dir}/{task}/{dataset}")]
                time = max(times)
                # to string
                time = time.strftime("%Y-%m-%d %H:%M:%S.%f")

                # train
                train_performance[task] = {}
                for epoch in range(20):
                    d = f"{dir}/{task}/{dataset}/{time}/train/{epoch}/{task}"
                    files = os.listdir(d)
                    for file in files:
                        with open(f"{d}/{file}", "rb") as f:
                            performance = pickle.load(f)

                            if task == 'classify':
                                file = file.split("_")[0]

                            if file not in train_performance[task]:
                                train_performance[task][file] = []
                            train_performance[task][file].append(performance[task])

                # 20 epochs
                assert len(train_performance[task][file]) == 20

                # convert to {key: []}
                train_performance[task][file] = {k: [train_performance[task][file][i][k] for i in range(20)] for k in train_performance[task][file][0]}
                train_performance[task][file]['best epoch'] = np.where(np.array(train_performance[task][file]['loss']) == min(train_performance[task][file]['loss']))[0][0]

                print(f"{task} {dataset} {file} \ntrain best epoch {train_performance[task][file]['best epoch']}", file=f)
                for metric in train_performance[task][file]:
                    if metric != 'best epoch':
                        print(metric, train_performance[task][file][metric][train_performance[task][file]['best epoch']], file=f)

                # test
                test_performance[task] = {}
                for epoch in range(20):
                    d = f"{dir}/{task}/{dataset}/{time}/test/{epoch}/{task}"
                    files = os.listdir(d)
                    for file in files:
                        with open(f"{d}/{file}", "rb") as f:
                            performance = pickle.load(f)

                            if task == 'classify':
                                file = file.split("_")[0]

                            if file not in test_performance[task]:
                                test_performance[task][file] = []
                            test_performance[task][file].append(performance[task])

                # 20 epochs
                # assert len(test_performance[task][file]) == 20

                # convert to {key: []}
                test_performance[task][file] = {k: [test_performance[task][file][i][k] for i in range(20)] for k in test_performance[task][file][0]}


                print("test", file=f)
                for metric in test_performance[task][file]:
                    print(metric, test_performance[task][file][metric][train_performance[task][file]['best epoch']], file=f)



    print()





def print_mpt_performance(dir, time=None):
    if time is None:
        # select latest
        times = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in os.listdir(dir)]
        time = max(times)
        # to string
        time = time.strftime("%Y-%m-%d %H:%M:%S.%f")

    # write to file
    with open(f"{dir}/performance.txt", "w") as f:

        train_performance = {}
        test_performance = {}
        tasks = ['anomaly', 'classify', 'forecasting_long', 'forecasting_short', 'imputation']

        # train
        for task in tasks:
            train_performance[task] = {}

            for epoch in range(20):
                d = f"{dir}/{time}/train/{epoch}"
                files = os.listdir(f"{d}/{task}")
                for file in files:
                    with open(f"{d}/{task}/{file}", "rb") as f:

                        if task == 'classify':
                            file = file.split("_")[0]


                        performance = pickle.load(f)
                        if file not in train_performance[task]:
                            train_performance[task][file] = []
                        train_performance[task][file].append(performance[task])

        # convert to {key: []}
        for task in tasks:
            for file in train_performance[task]:
                train_performance[task][file] = {k: [train_performance[task][file][i][k] for i in range(20)] for k in train_performance[task][file][0]}
                train_performance[task][file]['best epoch'] = np.where(np.array(train_performance[task][file]['loss']) == min(train_performance[task][file]['loss']))[0][0]

        # test
        for task in tasks:
            test_performance[task] = {}

            for epoch in range(20):
                d = f"{dir}/{time}/test/{epoch}"
                files = os.listdir(f"{d}/{task}")
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
                test_performance[task][file] = {k: [test_performance[task][file][i][k] for i in range(20)] for k in test_performance[task][file][0]}



        for task in tasks:
            for file in test_performance[task]:
                print(f"{task} {file} \ntrain best epoch {train_performance[task][file]['best epoch']}", file=f)
                for metric in train_performance[task][file]:
                    if metric != 'best epoch':
                        print(metric, train_performance[task][file][metric][train_performance[task][file]['best epoch']], file=f)

                print("test", file=f)
                for metric in test_performance[task][file]:
                    print(metric, test_performance[task][file][metric][train_performance[task][file]['best epoch']], file=f)

    print()








if __name__ == "__main__":

    print_mpt_performance("performance/MPT_prompttune", time='2024-07-26 22:57:30.864196')


    print_multivariable_performance("performance/prompttune_attention")
    print_multivariable_performance("performance/prompttune_linear")




    dir = "performance/MPT_finetune"
    time = '2024-07-25 13:35:13.858205'

    files = []

    performances = {}
    for partition in ['train', 'test']:
        performances[partition] = {}
        for epoch in os.listdir(f"{dir}/{time}/{partition}/"):
            epoch = int(epoch)
            performances[partition][epoch] = {}
            for task in os.listdir(f"{dir}/{time}/{partition}/{epoch}/"):
                performances[partition][epoch][task] = {}

                if task == 'anomaly':
                    for file in os.listdir(f"{dir}/{time}/{partition}/{epoch}/{task}"):
                        data_name = file.split("_")[0]
                        if data_name not in performances[partition][epoch][task]:
                            performances[partition][epoch][task][data_name] = []

                        with open(f"{dir}/{time}/{partition}/{epoch}/{task}/{file}", "rb") as f:
                            performance = pickle.load(f)
                            performances[partition][epoch][task][data_name].append(performance)

                    for data_name in performances[partition][epoch][task]:
                        f1s = []
                        losses = []
                        for i in range(len(performances[partition][epoch][task][data_name])):
                            f1s.append(performances[partition][epoch][task][data_name][i]['anomaly']['adj F1'])
                            losses.append(performances[partition][epoch][task][data_name][i]['anomaly']['loss'])
                            # TODO: what's F1 == 0?

                        performances[partition][epoch][task][data_name] = {}
                        performances[partition][epoch][task][data_name]['anomaly'] = {}
                        performances[partition][epoch][task][data_name]['anomaly']['adj F1'] = np.mean(f1s)
                        performances[partition][epoch][task][data_name]['anomaly']['loss'] = np.mean(losses)
                        files.append(data_name)

                else:
                    for file in os.listdir(f"{dir}/{time}/{partition}/{epoch}/{task}"):
                        with open(f"{dir}/{time}/{partition}/{epoch}/{task}/{file}", "rb") as f:
                            performance = pickle.load(f)
                            performances[partition][epoch][task][file] = performance
                            files.append(file)


    files = list(set(files))

    for file in files:
        fig, axs = plt.subplots(2, 1)
        for i, partition in enumerate(['train', 'test']):
            p = {'imputation': [], 'anomaly': [], 'classify': [], 'forecasting_long': [], 'forecasting_short': []}
            for epoch in os.listdir(f"{dir}/{time}/{partition}"):
                epoch = int(epoch)

                for task in os.listdir(f"{dir}/{time}/{partition}/{epoch}"):
                    f = f"{dir}/{time}/{partition}/{epoch}/{task}/{file}"
                    if os.path.exists(f):
                        with open(f, "rb") as f:
                            performance = pickle.load(f)
                            p[task].append(performance[task]['loss'])

            for task in p:
                if len(p[task]) > 0:
                    axs[i].plot(p[task], label=f"{partition}_{task}")
                    axs[i].set_title(file)
                    axs[i].set_xlabel("epoch")
                    axs[i].legend()

        if not os.path.exists(f"plots/{dir}"):
            os.makedirs(f"plots/{dir}")
        plt.savefig(f"plots/{dir}/{file}.png")
        plt.close()

