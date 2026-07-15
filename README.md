# 🚀 Multimodal Fake News Detection using Text and Image Fusion

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Dataset](https://img.shields.io/badge/Dataset-Fakeddit-green)

## Abstract

This project presents a **multimodal deep learning framework** for fake
news detection using the **Fakeddit** dataset. Unlike traditional
text-only systems, it combines **BERT (bert-base-uncased)** for textual
understanding and **ResNet-50** for visual feature extraction. The
extracted representations are fused to classify news as **REAL** or
**FAKE**.

------------------------------------------------------------------------

# Motivation

Fake news frequently combines misleading headlines with persuasive
images. This project jointly analyzes both modalities to improve
classification performance.

------------------------------------------------------------------------

# Dataset

Dataset: **Fakeddit**

Files used:

-   `multimodal_train.tsv`
-   `multimodal_validate.tsv`
-   `multimodal_test_public.tsv`

Columns used by this project:

-   `clean_title`
-   `title`
-   `image_url`
-   `hasImage`
-   `2_way_label`

Rows without images are filtered before training.

------------------------------------------------------------------------

# Methodology

1.  Download and cache images
2.  Tokenize text using **BERT Tokenizer**
3.  Extract text embeddings using **BERT**
4.  Extract image embeddings using **ResNet-50**
5.  Concatenate both feature vectors
6.  Classify using fully connected layers

------------------------------------------------------------------------

# Model Architecture

``` text
          Text
            │
            ▼
   BERT-base-uncased
            │
      CLS Embedding (768)

            +

          Image
            │
            ▼
        ResNet-50
            │
   Image Features (2048)

            │
     Feature Concatenation
            │
      Fusion Neural Network
            │
      REAL / FAKE
```

------------------------------------------------------------------------

# Training Features

-   AdamW Optimizer
-   Cross Entropy Loss
-   Mixed Precision (AMP)
-   Early Stopping
-   Gradient Clipping
-   Learning Rate Scheduler
-   Checkpoint Saving

------------------------------------------------------------------------

# Evaluation

The model reports:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   ROC-AUC
-   Classification Report
-   Confusion Matrix

------------------------------------------------------------------------

## Example 1

**Input Text**

PD: Phoenix car thief gets instructions from YouTube video

![Example 1](assets/example1.jpg)

Ground Truth: **FAKE**

Model Prediction: **FAKE**

---

## Example 2

**Input Text**

I do believe there's a squatch in these woods

![Example 2](assets/example2.jpg)

Ground Truth: **TRUE**

Model Prediction: **TRUE**

------------------------------------------------------------------------

# Project Structure

``` text
multimodal-fake-news/
├── config.py
├── download_images.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

------------------------------------------------------------------------

# Installation

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Usage

``` bash
python download_images.py
python main.py --mode train
python main.py --mode eval --checkpoint checkpoints/best_model.pt
python main.py --mode predict --checkpoint checkpoints/best_model.pt --title "News title" --image_url "IMAGE_URL"
```

------------------------------------------------------------------------

# Future Work

-   Vision Transformers (ViT)
-   CLIP-based multimodal learning
-   Real-time fake news detection
-   Web deployment

------------------------------------------------------------------------

# Author

**Digumurthy Sruthi Sarika**
