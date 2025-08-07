# Pneumonia Detection using Convolutional Neural Networks (CNN)

This deep learning project applies convolutional neural networks (CNNs) to classify chest X-ray images as either **Normal** or **Pneumonia**. The model is trained, validated, and evaluated using a well-known medical imaging dataset.

---

## Dataset

- **Source**: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Structure**: Contains chest X-ray images labeled as either `NORMAL` or `PNEUMONIA`
- **Splits**: Pre-divided into `train`, `val`, and `test` sets
- Added pictures to val from train and test for better training

---

## Objectives

- Build a CNN model to detect pneumonia in X-ray images
- Improve performance using data augmentation
- Evaluate model using accuracy, confusion matrix, and classification report

---

## Model Architecture

- 3 Convolutional layers with ReLU activation and MaxPooling
- Flatten layer followed by Dense layers
- Output layer with Softmax activation (for 2-class classification)
- Compiled with Adam optimizer and categorical cross-entropy loss

---

## Techniques Applied

- Data augmentation (rotation, zoom, shift, horizontal flip)
- Batch image processing using Keras `ImageDataGenerator`
- Visualization of training samples and performance metrics
- Model saving for reuse and deployment
- Added early stopping and model checkpoint callbacks

---

## Visualizations

- Sample augmented training images
- Confusion matrix on test set
- Classification report with precision, recall, F1-score

---

## Results

- **Test Accuracy**: ~83%
- **Performance**: Balanced precision/recall between NORMAL and PNEUMONIA classes
- **Saved Model**: `pneumonia_model.keras`

---

## What I Learned

- How to preprocess and augment image datasets for CNNs
- How to build and tune CNN architecture for medical image classification
- The importance of validation and evaluation on unseen data

---

## Future Improvements

- Experiment with transfer learning (e.g. VGG16, ResNet)
- Explore Grad-CAM for interpretability of CNN predictions

- **Credits**: Dataset originally published by Paul Mooney on Kaggle
