# BottleAROL

**BottleAROL** is a project for the detection and classification of bottle caps using **YOLOv11**. The goal of this project is to automatically detect and classify different types of bottle caps, which include:

- Wet/Dirty Sealed Caps
- Open Caps
- Caps without Rings
- Caps without Anything
- Broken Caps

The model is trained on a custom dataset, and the code is implemented using YOLOv11, an advanced one-stage object detection algorithm.

## Project Structure

The repository is organized as follows:

- `notebooks/`: Contains Jupyter notebooks for training and evaluation.
- `src/`: Contains the source code for the model, data preprocessing, and other utilities.
- `data/`: Custom dataset used for training and evaluation.

## Getting Started

To run the code locally, follow these steps to set up the environment on your machine.

### 1. Set Up the Virtual Environment

First, create a Conda environment using the provided `environment.yml` file:

```bash
# Create Conda environment
conda env create -f environment.yml

# Activate the virtual environment
conda activate ./env
```

If you need to update the environment with new dependencies, run the following command:

```bash
# Update the Conda environment
conda env update --file environment.yml --prune -p ./env
```

### 2. Install Dependencies

If any new dependencies are added later or if you want to install them manually, you can install them by running:

```bash
# Install dependencies listed in environment.yml
conda install --file environment.yml
```

### 3. Running the Code

Once the environment is set up, navigate to the `notebooks/` directory to find Jupyter notebooks for training and evaluation. You can run these notebooks to train the YOLOv11 model on the custom dataset.

### 4. Dataset

The dataset used in this project is custom and contains images of different bottle caps, categorized into the aforementioned classes. Ensure that you have the necessary permissions to access and use the dataset. You can adjust the paths and configurations in the notebook to point to your dataset.

## YOLOv11 Overview

YOLOv11 is an advanced version of the popular YOLO (You Only Look Once) algorithm for object detection. YOLOv11 is optimized for better accuracy and performance in real-time object detection tasks.

For more details on how to train and use YOLOv11 for custom tasks, refer to the official YOLOv11 documentation.


## Acknowledgements

- YOLOv11 for object detection.
- The dataset used in this project, made ad hoc for the task.
