
# Sinhala Character Recognition using KNN

This project implements a machine learning pipeline for recognizing Sinhala handwritten characters using the K-Nearest Neighbors (KNN) algorithm. It includes dataset creation, model training, and a simple GUI for testing predictions.

## Features
- **Dataset Creation:** Tools and scripts to collect and preprocess Sinhala character images.
- **Model Training:** Train a KNN classifier on the prepared dataset.
- **GUI Application:** Test handwritten character recognition using a graphical interface.

## Project Structure
- `1-dataset-creation.ipynb` – Notebook for creating and preprocessing the dataset.
- `2-training-the-KNN.ipynb` – Notebook for training the KNN model.
- `3-sinhala-character-gui.ipynb` – Notebook for running the GUI application.
- `data.npy`, `target.npy` – Numpy arrays containing image data and labels.
- `sinhala-character-knn.sav` – Saved KNN model.
- `data/`, `dataset/` – Folders containing raw and processed images.
- `MY/` – Additional scripts, notebooks, and resources.

## Getting Started
1. **Install Requirements**
	- Python 3.x
	- Install dependencies:
	  ```bash
	  pip install opencv-python numpy scikit-learn matplotlib
	  ```
2. **Create Dataset**
	- Run `1-dataset-creation.ipynb` to generate the dataset.
3. **Train Model**
	- Run `2-training-the-KNN.ipynb` to train and save the KNN model.
4. **Test with GUI**
	- Run `3-sinhala-character-gui.ipynb` to launch the GUI and test predictions.

## Usage
- You can add your own images to the `dataset/` folder, organized by character.
- The GUI allows you to draw or upload an image and see the predicted Sinhala character.

## Acknowledgements
- Developed by avishka-m
- Uses OpenCV, NumPy, and scikit-learn

---
Feel free to contribute or raise issues for improvements!
