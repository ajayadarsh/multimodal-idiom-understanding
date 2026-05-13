---
title: "CUB-200-2011 Fine-Grained Classification (Task 1)"
author: "Ajay Adarsh Sivakumar"
date: "17/02/2026"

---

A deep learning project for classifying 200 visually similar bird species using transfer learning and a custom-designed CNN. This project explores fine-grained classification challenges and demonstrates how modern techniques like augmentation, attention mechanisms, and test-time augmentation improve performance.

---

## Project Highlights

- Achieved **~80% test accuracy** on a **200-class** fine-grained dataset
- Implemented **EfficientNetB2 transfer learning (Model 1)**
- Designed a **custom CNN with channel attention (Model 2)**
- Applied advanced augmentation:
  - MixUp
  - CutMix (experimental)
- Used **Flip Test-Time Augmentation (TTA)** for improved predictions
- Built a **complete training + evaluation + demo pipeline**

---

## Results

### Model 1 – EfficientNetB2 (Transfer Learning)

| Metric        | Value |
|--------------|------|
| Accuracy     | 79.6% |
| Precision    | 0.808 |
| Recall       | 0.798 |
| F1 Score     | 0.797 |

---

### Model 2 – Custom CNN with Attention

| Metric        | Value |
|--------------|------|
| Accuracy     | ~80% |
| Precision    | ~0.84 |
| Recall       | ~0.80 |
| F1 Score     | ~0.80 |

---

## Confusion Matrix

![Confusion Matrix](Results/test_confusion_matrix.png)

---

## Approach

### Data Processing
- Bounding box cropping with padding (focus on bird region)
- Image resizing to **260×260**
- EfficientNet normalization
- Stratified train-validation split

---

### Data Augmentation
- Random flip, rotation, zoom, contrast, translation
- MixUp regularization to improve generalization

# Model 1 - EfficientNetB2 standard Model with Finetuning

## Requirements
- Python 3.x
- TensorFlow 2.x
- numpy, pandas, scikit-learn, matplotlib

## Dataset
Use the dataset provided on Canvas (CUB_200_2011).  
Do not train on the official test split.
CUB_200_2011/
    images/
    bounding_boxes.txt
    train_test_split.txt
    image_class_labels.txt
    classes.txt

## Reproducibility
- Random seed: 42
- Determinism enabled where supported (tf.config.experimental.enable_op_determinism)
- Mixed precision training enabled
- All checkpoints saved

## Project Structure

cub-bird-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── pipeline/
│   ├── data_pipeline.py
│   ├── model1.py
│   ├── model2.py
│
├── notebook/
│   ├── Bird_Classification_DL.py
│
├── results/
│   ├── test_confusion_matrix.npy
│   └── final_test_metrics.txt
│   └── test_predictions_with_names.txt
│
└── Report/
    └── Report - Fine Grained Bird Species Classification Final.pdf

## Model 1 - EfficientNetB2 (Standard Transfer Learning Model)

## Architecture
- Backbone: EfficientNetB2 pretrained on ImageNet
- Input size: 260 × 260
- Global Average Pooling
- Dropout: 0.30
- Dense Softmax Output (200 classes)

## Training Configuration
- Backbone: EfficientNetB2 pretrained on ImageNet
- Input size: 260x260
- Batch size: 16
- Bounding-box crop with padding = 0.15
- Augmentation: flip/rotation/zoom/contrast/translation (moderate)
- MixUp alpha: 0.1 (train only)
- Optimizer: AdamW (lr=1e-3, wd=1e-4) for head training
- Fine-tune: unfreeze last 40 layers, cosine-decay LR (~5e-6)

## How to run
1) Run the below notebook cells in order:
   - Load CUB-200-2011 Dataset into Compute instance storage
   - Load dataset and metadata, Train/Test Split and Bounding Boxes
   - Stratified Train/Validation Split
   - Build TensorFlow Data Pipeline (BBox Crop, Augmentation, Preprocessing)
   - MixUp Regularisation and One-Hot Label Preparation (Training Only)
   - Build EfficientNet-B2 Model and Train Classification Head (Transfer Learning)
   - Fine-tuning with Cosine Decay LR
   - Evaluate train(no-aug) and val
   - Validation Evaluation: Accuracy, Precision, Recall, F1 and Confusion Matrix (Flip-TTA)
   - Final Test Evaluation using Flip Test-Time Augmentation (TTA)
   - Save Final Model and Test Outputs to Google Drive (Checkpoint + Predictions + Confusion Matrix)
   - Save Per-Image Test Predictions to CSV
   - Plot Confusion Matrix

2) Best checkpoint is saved at:
   - /content/checkpoints/standard_efficientnetb2_finetuned_best_bbox.keras
   - Copied to Drive: /content/drive/MyDrive/cub_models/efficientnetb2_final.keras

# Model 2 – Custom Bird Classification Network

## Preprocessing

- Bounding box crop with padding
- Resize to 260x260
- EfficientNet-style normalization
- Stratified split
- tf.data pipeline with prefetching
- Random seed fixed for reproducibility

## Architecture

Custom CNN consisting of:

- Multiple convolutional blocks
- Batch Normalization
- Channel attention module
- Global Average Pooling
- Dropout (0.4)
- Dense softmax classifier

## Training Configuration

Optimizer: Adam  
Initial learning rate: 3e-4  
Fine-tuning learning rate: 1e-4  
Learning rate scheduling: ReduceLROnPlateau  
Mixed precision training enabled  
Early stopping used  
Best checkpoint saved during training  

Data augmentation:
- Random horizontal flip
- Random rotation
- Random contrast
- MixUp

## How to run
1) Run the below notebook cells in order:
   - Load CUB-200-2011 Dataset into Compute instance storage
   - Load dataset and metadata, Train/Test Split and Bounding Boxes
   - Stratified Train/Validation Split
   - Build TensorFlow Data Pipeline (BBox Crop, Augmentation, Preprocessing)
   - MixUp Regularisation and One-Hot Label Preparation (Training Only)
   - Build EfficientNet-B2 Model and Train Classification Head (Transfer Learning)
   - MODEL 2 (Custom design): ImageNet backbone + Custom Head
   - Plot TEST confusion matrix

2) Best checkpoint is saved at:
   - /content/checkpoints/model2_custom_finetuned_best.keras

## Demo
Path containing the demo folder:
Test/<class_folder>/<images>

Run the below cell which has the demo function:

DEMO: Evaluate Folder-Based Test Structure

It prints accuracy, precision, recall, F1, confusion matrix and per-image predictions.
