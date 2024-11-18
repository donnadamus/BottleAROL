"""
!gdown --id 1JIpccI_hh8L5Rr_yf81cSXA3__K0fXVn

!unzip /content/dataset_label.zip -d /content

# !unzip /content/drive/MyDrive/PROJECT\ SDP/dataset_label.zip -d /content

!pip install pytorch_lightning

"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from timm import create_model
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
from utils.task2_utils import *



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