# Project Title
Identify Hand-Drawn Illustrations

## Description
- We fine-tune a CNN model (ResNet18) to recognize hand-drawn grayscale images.
- The dataset is sourced from Google's Quick, Draw! project.
- For simplicity, we use only 20 classes for training.

## Getting Started

### Dependencies
- OS: Linux 22.04.5 LTS or higher
- Python 3.10.12 or higher
- NumPy 2.1.3 or higher
- PyTorch 2.6.0 or higher
- OpenCV 4.11.0.86 or higher
- tqdm 4.67.1 or higher
- scikit-learn 1.6.1 or higher

### Downloading the Dataset
- Link to the dataset: [Quick, Draw! Dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false&inv=1&invt=AbrzMQ)
- Download 20 arbitrary classes.

### Executing the Program
1. Install PyTorch (if not already installed):
```
pip install torch
```

2. Install NumPy(if not already installed):
```
pip install numpy
```

3. Install OpenCV(if not already installed):
```
pip install opencv-python
```

4. Install tqdm(if not already installed):
```
pip install tqdm
```
5. Install scikit-learn(if not already installed):
```
pip install scikit-learn
```
6. Run `train.py` to train the model. You can modify hyperparameters in the `get_args()` function.
7. Draw a grayscale image (e.g., using Microsoft Whiteboard or similar tools).
8. Save the image and update the `image_path` hyperparameter in `inference.py`.
9. Run `inference.py`.

## Sample Result
![Alt text](./sample_result.png)
