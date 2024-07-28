"""
load performance from files
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == "__main__":

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

