### EVALUATION CODE
import torch
import torch.nn as nn
from timm import create_model
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
from task2_utils import *


# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset and DataLoader
root_dir = "/Users/marcodonnarumma/Desktop/BottleAROL/data/test_set"  # Replace with your dataset path
dataset = BottleRotationDataset(root_dir=root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load a saved checkpoint
model = ViTRegression.load_from_checkpoint("/Users/marcodonnarumma/Desktop/BottleAROL/src/models/vit_tiny_rotation_epoch_epoch=100.ckpt")

import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

# Ensure the output directory exists
output_dir = "/Users/marcodonnarumma/Desktop/BottleAROL/data/predictions_task2"
os.makedirs(output_dir, exist_ok=True)

# Function to make predictions, save images with labels, and return predictions and targets
def predict_and_save_images(model, dataloader, output_dir):
    model.eval()  # Set the model to evaluation mode
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_predictions = []  # Store all predictions
    all_targets = []      # Store all corresponding targets

    # Iterate through the DataLoader
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            predictions = model(images).squeeze()  # Get model predictions

        # Append predictions and targets to the lists
        all_predictions.extend(predictions.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

        # Process each image in the batch
        for i in range(len(images)):
            # Convert the image back to [0, 1] range
            image = images[i].cpu().permute(1, 2, 0).numpy()  # Convert to (H, W, C)
            image = (image * 0.5 + 0.5) * 255  # Reverse normalization and scale to 0-255
            image = Image.fromarray(image.astype("uint8"))

            # Create a new canvas with extra space above the image
            new_height = image.height + 50  # Add 50px for text
            canvas = Image.new("RGB", (image.width, new_height), "white")  # White background
            canvas.paste(image, (0, 50))  # Paste the image below the text area

            # Draw real and predicted labels on the canvas
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()  # Use a default font
            real_label = f"Real: {labels[i].item():.2f}"
            predicted_label = f"Pred: {predictions[i].item():.2f}"

            # Write the labels with different colors
            draw.text((10, 10), real_label, fill="red", font=font)  # Real label in red
            draw.text((10, 30), predicted_label, fill="blue", font=font)  # Predicted label in blue

            # Save the annotated image
            canvas.save(os.path.join(output_dir, f"image_{batch_idx}_{i}.png"))

    print(f"Annotated images saved to: {output_dir}")
    return all_predictions, all_targets  # Return all predictions and targets

# Run predictions, save images, and get predictions/targets
predictions, targets = predict_and_save_images(model, dataloader, output_dir)

# Print a few predictions and targets for verification
for i in range(5):  # Display the first 5 predictions and targets
    print(f"Target: {targets[i]:.2f}, Prediction: {predictions[i]:.2f}")

import numpy as np

def compute_metrics(predictions, targets):
    # Convert to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)

    # MAE
    mae = np.mean(np.abs(predictions - targets))

    # MSE
    mse = np.mean((predictions - targets) ** 2)

    # RMSE
    rmse = np.sqrt(mse)

    # MAPE
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100  # Avoid division by zero

    # R-squared
    target_mean = np.mean(targets)
    r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - target_mean) ** 2)

    # Angular Error
    angular_error = np.mean(np.minimum(
        np.abs(targets - predictions),
        360 - np.abs(targets - predictions)
    ))

    return {
        "Mean Absolute Error": mae,
        "Root Mean Squared Error": rmse,
    }

metrics = compute_metrics(predictions, targets)

for metric, value in metrics.items():
    print(f"{metric}: {value:.2f}")


"""

	•	MAE averages the errors linearly.
	•	RMSE grows quadratically due to squaring, amplifying the effect of the larger error.

"""