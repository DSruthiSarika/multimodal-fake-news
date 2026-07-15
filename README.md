# 🚀 Multimodal Fake News Detection using BERT and ResNet-50

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

This project uses the **Fakeddit** multimodal dataset for fake news detection.

Due to the large size of the dataset, it is **not included** in this repository.

## 📥 Download Dataset

Google Drive Folder:

https://drive.google.com/drive/folders/1nZ-FOqfJLZgjFB6jpd1hBXCuBi0vPaLS?usp=drive_link

The folder contains:

- `multimodal_train.tsv`
- `multimodal_validate.tsv`
- `multimodal_test_public.tsv`

Place all three files in the project root directory:

```text
multimodal-fake-news/
│
├── multimodal_train.tsv
├── multimodal_validate.tsv
├── multimodal_test_public.tsv
│
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

After placing the dataset files, download and cache the images:

```bash
python download_images.py
```

> **Note:** This project uses the **Fakeddit** dataset for academic and research purposes.
------------------------------------------------------------------------

## Preparing Images

Before training, download and cache the images:

```bash
python download_images.py
```

This step downloads all images referenced by the dataset and stores them locally for faster training.

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

Ground Truth: **REAL**

Model Prediction: **REAL**

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

## Requirements

- Python 3.10+
- PyTorch 2.x
- CUDA (optional, recommended)

------------------------------------------------------------------------

# Usage

``` bash
python download_images.py
python main.py --mode train
python main.py --mode eval --checkpoint checkpoints/best_model.pt
python main.py --mode predict --checkpoint checkpoints/best_model.pt --title "News title" --image_url "IMAGE_URL"
```

------------------------------------------------------------------------

## Output

During training, the project generates:

- `checkpoints/` — saved model checkpoints
- `image_cache/` — downloaded images
- Evaluation metrics
- Confusion matrix
- Classification report

------------------------------------------------------------------------

# Future Work

- Integrate Vision Transformers (ViT) for improved image representation.
- Explore CLIP-based multimodal feature learning.
- Extend the framework for multi-class fake news classification.
- Deploy the model as a web application using Streamlit or Flask.

------------------------------------------------------------------------

## License

This project is developed for academic and educational purposes.

------------------------------------------------------------------------

# Author

**Digumurthy Sruthi Sarika**
