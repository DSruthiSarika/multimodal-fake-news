# Multimodal Fake News Detection using Text and Image Fusion

## Overview

With the rapid growth of digital media, the spread of misleading and false information has become a major concern. Detecting fake news is challenging because such content often combines convincing text with misleading visuals.

This project explores a multimodal approach to fake news detection by combining both textual and visual information. Instead of relying only on text, the system integrates image-based features to better capture the context of news content and improve prediction performance.

---

## Problem Statement

Fake news detection is difficult due to:

* Highly unstructured and noisy data
* Nonlinear relationships between text and visual content
* Misleading headlines and fabricated claims
* Limitations of traditional text-only models

Most existing approaches focus only on textual analysis, which can miss important signals present in images.

---

## Proposed Solution

This project introduces a multimodal machine learning pipeline that combines:

* **Text Features**: Extracted using TF-IDF vectorization to capture key patterns in news content
* **Image Features**: Extracted from image URLs and converted into numerical representations
* **Feature Fusion**: Combines text and image features into a unified representation
* **Model Comparison**: Multiple models are trained and evaluated to select the best-performing approach

---

## Dataset

The dataset used in this project is available in the `data/` folder:

- final_multimodal_dataset.csv

It contains:
- Text data
- Image URLs
- Labels (REAL / FAKE)
- Source information
- Confidence scores

The dataset is intentionally lightweight and structured to demonstrate the effectiveness of multimodal learning.

---

## Methodology

The workflow of the system is as follows:

1. Load and preprocess the dataset
2. Convert textual data into numerical features using TF-IDF
3. Extract image features from URLs using OpenCV
4. Combine text and image features using feature fusion
5. Train multiple machine learning models:

   * Logistic Regression
   * Random Forest
   * XGBoost
6. Evaluate models using:

   * Accuracy
   * Classification report
   * Confusion matrix
7. Select and save the best-performing model

---

## Results

The multimodal approach shows improved performance compared to single-modality models. By combining both text and image information, the system is able to better capture complex patterns in the data.

Model comparison helps identify the most suitable algorithm, and visualization tools such as confusion matrices provide deeper insights into prediction performance.

---

## Project Structure

The project is organized as follows:

```bash
multimodal-fake-news/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── final_multimodal_dataset.csv
│
├── models/
│   └── best_model.pkl
│
├── assets/
│   ├── confusion_matrix.png
│   └── accuracy_plot.png
```

---

## How to Run the Project

### Step 1: Install dependencies

pip install -r requirements.txt

### Step 2: Run the project

python main.py

---

## Key Features

* Multimodal learning using both text and image data
* Feature fusion for improved prediction accuracy
* Comparison of multiple machine learning models
* Visualization of results through graphs and confusion matrices
* Clean and reproducible pipeline

---

## Applications

* Fake news detection systems
* Social media monitoring
* Content moderation platforms
* Media verification tools

---

## Conclusion

This project demonstrates how combining textual and visual data can improve the detection of fake news. The multimodal approach provides a more comprehensive understanding of content compared to traditional methods.

Although the current implementation uses a synthetic dataset, the same framework can be extended to real-world datasets and more advanced deep learning models.

---

## Note

This project uses a synthetic dataset created for experimental and demonstration purposes. It is designed to simulate real-world conditions and highlight the effectiveness of multimodal learning techniques.

---

## Author

Digumurthy Sruthi Sarika
