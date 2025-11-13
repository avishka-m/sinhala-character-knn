# Sinhala Character Recognition using KNN

A machine learning project that recognizes Sinhala handwritten characters using the K-Nearest Neighbors (KNN) algorithm. The project includes complete dataset creation, model training, and a graphical user interface for real-time character prediction.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

- **Dataset Creation**: Automated preprocessing of Sinhala character images
- **KNN Model Training**: Machine learning pipeline using scikit-learn
- **Interactive GUI**: Real-time handwriting recognition with Tkinter
- **Multi-character Support**: Recognition of අ (a), එ (ae), ඉ (e), උ (u) characters
- **Image Processing**: OpenCV-based image preprocessing and normalization

## 📁 Project Structure

```
├── dataset_creation.ipynb    # Dataset preprocessing and creation
├── train_knn.ipynb         # KNN model training and evaluation
├── gui.ipynb               # Interactive GUI application
├── data.npy                # Processed image data (64-dimensional vectors)
├── target.npy              # Character labels
├── sinhala_char_knn.sav    # Trained KNN model
├── dataset/                # Character image datasets
│   ├── a/                  # අ character images (70+ samples)
│   ├── ae/                 # එ character images (60+ samples)
│   ├── e/                  # ඉ character images
│   └── u/                  # උ character images
└── data/                   # Sample images and test data
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or VS Code with Jupyter extension

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/avishka-m/sinhala-character-knn.git
   cd sinhala-character-knn
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python numpy scikit-learn matplotlib pillow tkinter joblib
   ```

### Usage

#### Step 1: Dataset Creation
Run `dataset_creation.ipynb` to:
- Load and preprocess character images from the `dataset/` folder
- Resize images to 8x8 pixels for feature extraction
- Convert to grayscale and flatten to 64-dimensional vectors
- Save processed data as `data.npy` and `target.npy`

#### Step 2: Train the Model
Run `train_knn.ipynb` to:
- Load the preprocessed dataset
- Split data into training and testing sets (80/20)
- Train a KNN classifier
- Evaluate model accuracy
- Save the trained model as `sinhala_char_knn.sav`

#### Step 3: Use the GUI
Run `gui.ipynb` to launch the interactive application:
- **Draw**: Use mouse to draw characters on the white canvas
- **PREDICT**: Get real-time prediction of the drawn character
- **SAVE**: Save drawn images to the `data/` folder
- **CLEAR**: Clear the canvas for a new drawing

## 🧠 Model Details

- **Algorithm**: K-Nearest Neighbors (KNN) with default parameters
- **Features**: 8x8 grayscale pixel intensity values (64 features)
- **Classes**: 4 Sinhala characters mapped as:
  - අ (a) → Label 0
  - එ (ae) → Label 1  
  - ඉ (e) → Label 2
  - උ (u) → Label 3
- **Input Processing**: Images resized to 8x8, normalized, and flattened

## 📊 Dataset

The dataset contains handwritten samples of 4 Sinhala vowels:
- **අ (a)**: ~70 images (jpg/png format)
- **එ (ae)**: ~60 images (jpg/png format)  
- **ඉ (e)**: Various samples
- **උ (u)**: Various samples

Images are preprocessed to 8x8 grayscale format for consistent feature extraction.

## 🖥️ GUI Interface

The Tkinter-based GUI provides:
- **Canvas**: 500x500 pixel drawing area
- **Controls**: 
  - SAVE: Store drawings as samples
  - PREDICT: Classify the drawn character
  - CLEAR: Reset the canvas
  - EXIT: Close the application
- **Real-time Feedback**: Displays predicted character in Sinhala script

## 🔧 Technical Implementation

### Image Processing Pipeline
1. Load images using OpenCV
2. Convert to grayscale
3. Resize to 8x8 pixels
4. Flatten to 1D array (64 features)
5. Normalize pixel values

### Model Training
1. Load preprocessed data
2. Train/test split (80/20)
3. Fit KNN classifier
4. Evaluate performance
5. Save model using joblib

## 🤝 Contributing

Contributions are welcome! You can help by:
- Adding more character classes
- Improving model accuracy
- Enhancing the GUI interface
- Adding data augmentation techniques

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Avishka Maduranga** - [@avishka-m](https://github.com/avishka-m)

## 🙏 Acknowledgments

- Built with Python, OpenCV, scikit-learn, and Tkinter
- Inspired by handwritten character recognition research
- Thanks to the open-source community for the tools and libraries