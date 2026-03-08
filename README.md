# Food / Not Food Text Classification

A complete NLP text classification project that fine tunes DistilBERT to classify whether a sentence is about **food** or **not food**. This project covers the full workflow from dataset loading and preprocessing to model training, evaluation, Hugging Face Hub deployment, and a Gradio demo for interactive inference.

## Repository

`AnubhavKarki/hf-food-not-food-classification`

## Overview

This project builds a binary text classifier using the Hugging Face ecosystem. The model is trained on a dataset of food and not food image captions and learns to predict whether a given text input belongs to the `food` or `not_food` class.

The pipeline includes:

* loading a dataset from Hugging Face Datasets
* preprocessing and label encoding
* tokenization with DistilBERT tokenizer
* fine tuning a Transformer model for sequence classification
* evaluation on a held out test split
* saving and publishing the trained model to Hugging Face Hub
* building an interactive Gradio app
* preparing the app for deployment on Hugging Face Spaces

## Demo of the project in Gradio

![Demo of the project in Gradio](demo.png)

## Problem Statement

Given a text input such as a caption, phrase, or sentence, the goal is to classify it into one of two categories:

* `food`
* `not_food`

Examples:

* `"Salmon and rice is a healthy food combination."` → `food`
* `"Homage to Catalonia is written by George Orwell."` → `not_food`

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Evaluate
* Hugging Face Hub
* Gradio
* Hugging Face Spaces
* NumPy
* Pandas
* Matplotlib

## Dataset

The project uses the following dataset from Hugging Face:

`mrdbourke/learn_hf_food_not_food_image_captions`

This dataset contains text captions labeled as either food or not food.

### Dataset workflow

* load dataset from Hugging Face
* extract unique labels
* create `id2label` and `label2id` mappings
* convert labels into model ready numeric targets
* split into training and test sets using an 80/20 split

## Model

The base model used for fine tuning is:

`distilbert/distilbert-base-uncased`

### Why DistilBERT

DistilBERT provides a strong balance between performance and efficiency. It is smaller and faster than BERT while still retaining strong language understanding capabilities, making it a practical choice for text classification tasks.

## Project Pipeline

### 1. Data Loading

The dataset is loaded directly from Hugging Face using `datasets.load_dataset()`.

### 2. Label Mapping

String labels are mapped into numeric form for training.

Example mapping structure:

* `food` → `1`
* `not_food` → `0`

The exact mapping is created programmatically from the dataset.

### 3. Train Test Split

The dataset is shuffled and split into:

* 80% training data
* 20% test data

A fixed random seed is used for reproducibility.

### 4. Tokenization

Text inputs are tokenized using `AutoTokenizer` from the DistilBERT checkpoint.

Tokenization includes:

* padding
* truncation
* batching for faster preprocessing

