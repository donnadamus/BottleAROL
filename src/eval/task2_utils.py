import torch
import torch.nn as nn
from timm import create_model
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import pytorch_lightning as pl


# Define the dataset
class BottleRotationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        for degree_folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, degree_folder)
            if os.path.isdir(folder_path):
                degree = float(degree_folder)
                for img_name in os.listdir(folder_path):
                    self.image_paths.append(os.path.join(folder_path, img_name))
                    self.labels.append(degree)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# Define the custom PyTorch Lightning model
class ViTRegression(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.vit = create_model('vit_tiny_patch16_224', pretrained=True)
        num_features = self.vit.head.in_features
        self.vit.head = nn.Identity()  # Remove classification head
        self.regressor = nn.Linear(num_features, 1)  # Add regression head
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.vit(x)
        x = self.regressor(x)
        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch  # Targets are now in the range [0, 360]
        outputs = self(images).squeeze()  # Predictions from the model
        loss = self.criterion(outputs, targets)  # Compute loss without normalization

        # Log the learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=False, on_epoch=True)

        # Log only at the end of the epoch
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # Define default values
        default_learning_rate = 1e-4
        default_scheduler_patience = 5
        default_scheduler_factor = 0.1
        default_scheduler_threshold = 100  # Adjust to match your loss scale
        default_min_lr = 1e-6

        # Retrieve hyperparameters or use defaults
        learning_rate = self.hparams.get('learning_rate', default_learning_rate)
        scheduler_patience = self.hparams.get('scheduler_patience', default_scheduler_patience)
        scheduler_factor = self.hparams.get('scheduler_factor', default_scheduler_factor)
        scheduler_threshold = self.hparams.get('scheduler_threshold', default_scheduler_threshold)
        min_lr = self.hparams.get('min_lr', default_min_lr)

        # Define optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Define scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',                  # Minimize the monitored metric
            factor=scheduler_factor,     # Reduce learning rate by this factor
            patience=scheduler_patience, # Wait this many epochs with no improvement
            threshold=scheduler_threshold,  # Minimum improvement to consider
            threshold_mode='abs',        # Use absolute threshold for large-scale losses
            cooldown=0,                  # No cooldown period after reduction
            min_lr=min_lr                # Minimum learning rate allowed
        )

        # Return optimizer and scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',  # Metric to monitor for learning rate adjustment
                'interval': 'epoch',      # Check at the end of every epoch
                'frequency': 1            # Frequency of scheduler updates
            }
        }