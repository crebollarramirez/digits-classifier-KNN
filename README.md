# Digits Classification Using K-Nearest Neighbors (KNN)

This project demonstrates the implementation of a K-Nearest Neighbors (KNN) algorithm for classifying digits from the MNIST dataset and explores its application on other datasets, such as the Wine dataset. Both Scikit-learn’s KNN implementation and a custom-built KNN model are used to evaluate performance, hyperparameter tuning, and the impact of normalization and distance metrics.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Setup](#setup)
5. [Results](#results)

---

## Project Overview

The goal of this project is to classify handwritten digits from the MNIST dataset using the KNN algorithm. Key components include:

- Implementing KNN from scratch.
- Comparing with Scikit-learn's KNN.
- Optimizing hyperparameters (`k` and distance metrics).
- Evaluating the impact of normalization and scaling.
- Visualizing misclassifications to gain insights.

Additionally, the Wine dataset is used to test KNN on a classification problem with normalized features.


### Detailed Report
You can access the comprehensive analysis by clicking here: [Detailed Report](./Digits%20Classifier%20KNN.pdf)

---

## Features

- **Custom KNN Implementation**:

  - Implements KNN manually to understand the algorithm’s mechanics.
  - Supports different distance metrics, including Euclidean and cosine distances.

- **Scikit-learn KNN**:

  - Benchmarking against Scikit-learn’s optimized KNN.

- **Hyperparameter Tuning**:

  - Optimization of the number of neighbors (`k`) and distance metrics.

- **Data Visualization**:

  - Plotting misclassified digits and confusion matrices for deeper analysis.

- **Cross-Dataset Application**:
  - Testing KNN on a non-image dataset (Wine dataset) to showcase versatility.

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Scikit-learn
  - NumPy
  - Matplotlib
  - Pandas
- **Dataset**:
  - MNIST (subset of digits)
  - Wine dataset (from Scikit-learn)

---

## Setup

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Project**:
   Use Jupyter Notebook (`jupyter notebook`) to open and run the project.

---

## Results

### 1. **MNIST Dataset**

- **Scikit-learn KNN**:

  - Achieved an accuracy of **91.2%** with `k=5` and Euclidean distance.
  - Accuracy improved to **92.2%** using cosine distance.

- **Custom KNN Implementation**:

  - Achieved a slightly higher accuracy of **91.4%** with `k=5` and Euclidean distance.
  - Cosine distance also improved accuracy, matching Scikit-learn KNN at **92.2%**.

- **Impact of Normalization**:

  - Normalizing the dataset improved accuracy significantly for both Euclidean and cosine distances.

- **Misclassification Insights**:
  - Analyzed the most commonly misclassified digits:
    - Misclassified pairs often included visually similar digits (e.g., 8 and 3, 5 and 6).
  - Highlighted areas for improvement, such as feature extraction or weighted voting in KNN.

### 2. **Wine Dataset**

- **Without Normalization**:

  - Accuracy was **71%**, highlighting the impact of differing feature scales in the dataset.

- **With Normalization**:
  - Accuracy increased dramatically to **97%**, showcasing the importance of feature scaling in KNN.

### 3. **Hyperparameter Tuning**

- Optimized the number of neighbors (`k`) and distance metrics:
  - Increasing `k` reduced noise but slightly lowered accuracy after a certain point.
  - Euclidean distance worked well after normalization, while cosine distance performed consistently across datasets.

### 4. **Visualization and Metrics**

- **Confusion Matrices**:

  - Generated confusion matrices for both datasets to identify patterns in misclassifications.
  - Misclassification rates were concentrated in a few specific digit or class pairs.

- **Visualization**:
  - Plotted samples of misclassified digits to better understand challenges in the dataset.
  - Highlighted the effectiveness of cosine distance for visually complex digits.

---

These results demonstrate the effectiveness of the KNN model for classification tasks and emphasize the importance of normalization, distance metrics, and careful hyperparameter tuning.
