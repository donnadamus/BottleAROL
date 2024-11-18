import pytorch_lightning as pl
import torch
import torch.nn as nn
from timm import create_model
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
from src.utils.task2_utils import *


# Dataset and DataLoader
root_dir = "/content/dataset_label"  # Replace with your dataset path
dataset = BottleRotationDataset(root_dir=root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Lightning Trainer with Model Checkpoint Callback
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath="checkpoints/",
    filename="vit_tiny_rotation_epoch_{epoch}",
    save_top_k=-1,  # Save all checkpoints
    every_n_epochs=5,  # Save every 5 epochs
)

from pytorch_lightning.callbacks import ProgressBar

# Custom ProgressBar to ensure logging at the end of the epoch
class CustomProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss", None)
        learning_rate = trainer.callback_metrics.get("learning_rate", None)
        if train_loss:
            print(f"Epoch {trainer.current_epoch + 1}: Train Loss = {train_loss:.4f}")

        if learning_rate:
            print(f"Epoch {trainer.current_epoch + 1}: Learning Rate: {learning_rate:.6f}")
            
# Trainer configuration
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    callbacks=[checkpoint_callback, CustomProgressBar()],  # Add the custom ProgressBar
)

# Initialize and Train
model = ViTRegression.load_from_checkpoint("/content/vit_tiny_rotation_epoch_epoch=100.ckpt")
trainer.fit(model, dataloader)