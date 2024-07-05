import torch
import pandas as pd
import zipfile
from Dataset import CORAADisfluencyDataset
from Model import DisfluencyModel, Wav2Vec_DisfluencyModel
from Trainer import Trainer
import os
from datetime import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
METADATA_TEST = '/workspace/biometria/Datasets/CORAA/_metadata_/metadata_test_final.csv'
METADATA_TRAIN = '/workspace/biometria/Datasets/CORAA/_metadata_/metadata_train_final.csv'
metadata_train = pd.read_csv(METADATA_TRAIN)
metadata_test = pd.read_csv(METADATA_TEST)

data_dir = '/workspace/biometria/Datasets/CORAA/'
train_dataset = CORAADisfluencyDataset(metadata_train, data_dir)
print(f"Total train files: {len(train_dataset)}")
test_dataset = CORAADisfluencyDataset(metadata_test, data_dir)
print(f"Total validation files: {len(test_dataset)}")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    num_workers=8,
    batch_size=12
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16
)

CONFIG = {
    "max_epochs": 300, 
    "log_interval_updates": 5,
    "save_interval_updates": 100,
    "keep_interval_updates": 3,
    "max_updates": None,
    "device": DEVICE,
    "random_seed": 1,
    "optimizer": {
        "lr": 0.0001
    }
}

model = Wav2Vec_DisfluencyModel()

trainer = Trainer(
    model,
    train_loader,
    test_loader,
    CONFIG,
    log_dir="logs_2/"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

last_checkpoint_path = "logs_2/2024-06-19_02-36-20/checkpoint_last.pt"
if os.path.exists(last_checkpoint_path):
    trainer.load_checkpoint(last_checkpoint_path)
else:
    print(f"Nenhum checkpoint encontrado em {last_checkpoint_path}. Iniciando treinamento do zero.")



trainer.train()
