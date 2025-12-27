# 🐱🐶 Cat vs Dog Image Classification using HOG and SVM

This project implements a **Cat vs Dog image classifier** using classical
computer vision and machine learning techniques.
The model uses **HOG (Histogram of Oriented Gradients)** and **RGB color histograms**
for feature extraction and an **SVM with RBF kernel** for classification.

---

## 🔍 Feature Extraction

### 1️⃣ Histogram of Oriented Gradients (HOG)
- Captures **edge directions and shape information**
- Image is divided into small cells
- For each cell, gradient directions are computed and stored in histograms

### 2️⃣ RGB Color Histograms
- Captures **color distribution** in the image
- For each channel (Red, Green, Blue), pixel intensities are divided into bins
- Counts how many pixels fall into each intensity range

The final feature vector is created by **combining HOG features with RGB color histograms**.

---

## 🧠 Model Used

- Support Vector Machine (SVM)
- Kernel: **RBF (Radial Basis Function)**
- Suitable for **non-linear classification**

---

