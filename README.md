# Multimodal Fake News Detection using Text and Image Fusion

## Overview

The rapid spread of misinformation across digital platforms has made fake news detection an important problem in today’s information-driven world. Most existing systems rely only on textual data, which limits their ability to capture the full context of a news article.

This project presents a multimodal approach that combines both textual and visual information to improve the detection of fake news. By integrating features extracted from news content and associated images, the model aims to provide more accurate and reliable predictions.

---

## Problem Statement

Detecting fake news is challenging due to the presence of misleading content, sensational claims, and the complex relationship between text and visual elements. Traditional models often struggle because they:

* Depend only on textual data
* Fail to capture visual context
* Perform poorly on highly variable and noisy data

This project addresses these limitations by combining multiple data modalities into a unified system.

---

## Proposed Approach

The system follows a hybrid pipeline that integrates Natural Language Processing and basic Computer Vision techniques.

* Text data is processed using TF-IDF vectorization to capture important keywords and patterns
* Image data is converted into numerical features using pixel-based representations
* Both feature sets are combined to form a unified input for classification
* Multiple machine learning models are trained and compared to identify the best-performing approach

---

## Dataset

The project uses a synthetic dataset designed to simulate real-world news scenarios. It contains:

* News text
* Image URLs associated with each news item
* Labels indicating whether the news is real or fake
* Additional attributes such as source type and confidence level

The dataset is intentionally lightweight and structured to demonstrate the effectiveness of multimodal learning.

---

## Methodology

The overall workflow of the system includes:

1. Data loading and preprocessing
2. Text feature extraction using TF-IDF
3. Image feature extraction from URLs
4. Feature fusion to combine text and image data
5. Training multiple models including Logistic Regression, Random Forest, and XGBoost
6. Evaluating performance using accuracy, classification reports, and confusion matrices

---

## Results

The multimodal approach demonstrates improved performance compared to single-modality methods. The use of multiple models allows for comparison, and the best-performing model is selected based on accuracy.

Visualizations such as confusion matrices and accuracy comparisons help in understanding model behavior and performance.

---

## How to Run the Project

### Step 1: Install dependencies

pip install -r requirements.txt

### Step 2: Run the script

python main.py

---

## Project Structure

multimodal-fake-news/
│── main.py
│── README.md
│── requirements.txt
│
├── data/
│   ├── final_multimodal_dataset.csv
│
├── models/
│   ├── best_model.pkl
│
├── assets/
│   ├── results.png

---

## Key Features

* Multimodal learning using text and image data
* Feature fusion for improved prediction
* Comparison of multiple machine learning models
* Visualization of results using confusion matrices and graphs
* Clean and reproducible pipeline

---

## Applications

* Social media content monitoring
* Fake news detection platforms
* Content moderation systems
* Media verification tools

---

## Conclusion

This project demonstrates how combining textual and visual information can improve fake news detection. The system effectively integrates different data sources and uses machine learning models to generate reliable predictions.

While the current implementation uses a simplified dataset, the approach can be extended to real-world applications with larger and more complex datasets.

---

## Note

This project uses a synthetic dataset created for experimental and demonstration purposes. It is designed to simulate real-world scenarios and highlight the effectiveness of multimodal learning techniques.
