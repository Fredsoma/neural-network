# Traffic Sign Recognition Project

## Overview

This project uses a neural network to recognize traffic signs. It includes:
- **traffic.py**: A script for training the model.
- **best_model.h5**: The saved model generated after training.
- **predict_sign.py**: A Tkinter GUI that lets users upload an image and receive a prediction.
- **images/**: A folder containing 10 sample traffic sign images (one image per category, named with the category number).

## Setup Instructions

1. **Install Dependencies:**

   Ensure you have installed the required libraries. From the environment where TensorFlow is installed, run:
   pip install scikit-learn opencv-python

2. **Prepare the Dataset:**

Organize your training data into subfolders (named by category, e.g., `0`, `1`, etc.) and update the `dataset_dir` variable in `traffic.py` with the correct path.

3. **Train the Model:**

Run the training script: python traffic.py
The best model will be saved as `best_model.h5`.

4. **Test the Predictor:**

Launch the GUI by running: python predict_sign.py
Use the file dialog to select an image and see the prediction.


