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



def print_multivariable_performance(dir):
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





def print_mpt_performance(dir, time=None):
    if time is None:
        # select latest
        times = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in os.listdir(dir)]
        time = max(times)
        # to string
        time = time.strftime("%Y-%m-%d %H:%M:%S.%f")

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








if __name__ == "__main__":

    print_mpt_performance("performance/MPT_prompttune", time='2024-07-31 10:59:58.071013')
    print_mpt_performance("performance/MPT_finetune", time='2024-07-31 10:59:58.100640')

    print_multivariable_performance("performance/prompttune_attention")
    print_multivariable_performance("performance/finetune")
    print_multivariable_performance("performance/prompttune_linear")



