import os
os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"

from datasets import get_dataset_config_names, load_dataset
import huggingface_hub


from src.data import AnomalyDetectionDatasetMultiFile



# huggingface_hub.list_files_info("AutonLab/Timeseries-PILE")


# dataset1 = load_dataset("AutonLab/Timeseries-PILE", data_files="classification/UCR/ACSF1/ACSF1_TEST.ts",)
dataset = load_dataset("AutonLab/Timeseries-PILE", data_files="anomaly_detection/TSB-UAD-Public/Daphnet/S01R02E0.test.csv@1.out",)
# dataset = load_dataset("AutonLab/Timeseries-PILE", data_files="forecasting/autoformer/ETTh1.csv",)




# configs = get_dataset_config_names("AutonLab/Timeseries-PILE")


# from huggingface_hub import HfFileSystem

# fs = HfFileSystem()
# fs.ls("AutonLab/Timeseries-PILE")

