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

