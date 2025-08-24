📖 Persian BERT Text Classification

A complete end-to-end student project: dataset → fine-tuning → evaluation → Hugging Face Hub

✨ Project Overview

This project demonstrates how to fine-tune a Persian BERT model for text classification (e.g., sentiment/emotion analysis).
It is designed as a student-friendly tutorial, showing the complete workflow from data preparation to model deployment.

With this tutorial, you will learn how to:

✅ Load and preprocess Persian text data

✅ Encode labels for classification

✅ Fine-tune ParsBERT (HooshvareLab/bert-fa-zwnj-base) on your dataset

✅ Evaluate your model with accuracy, precision, recall, F1

✅ Visualize results with confusion matrix & confidence plots

✅ Save and upload your model to the Hugging Face Hub

✅ Run inference with new Persian sentences

🛠️ Requirements

Install dependencies:

pip install transformers datasets accelerate evaluate scikit-learn seaborn huggingface_hub

📂 Project Structure
persian-bert-text-classification/
│── dataset.csv              # Your dataset (text + label)
│── training_script.ipynb    # Main Colab/Jupyter script
│── results/                 # Training outputs (checkpoints, logs)
│── persian-bert-emotion/    # Saved fine-tuned model

📊 Dataset Format

Your dataset must be a CSV file with two columns:

text	label
امروز روز خوبی است	positive
من خیلی ناراحتم	negative
این فیلم عالی بود	positive
چقدر خسته‌ام	negative
🚀 Training

Run the notebook step by step:

Setup (libraries, GPU check, seed)

Load Dataset (CSV file with text + label)

Preprocessing (normalize Persian text)

Tokenization (ParsBERT tokenizer)

Model Setup (classification head + label mapping)

Training (fine-tune with Trainer)

Evaluation (metrics + visualization)

Save & Upload (to Hugging Face Hub)

Inference Demo (test sentences in Persian)

📈 Evaluation Visuals

The notebook generates:

🎯 Confusion Matrix

📊 F1-Score per class

📈 Precision vs Recall scatter plot

🔍 Confidence distribution (correct vs incorrect predictions)

🌐 Upload to Hugging Face Hub

Login to your Hugging Face account:

from huggingface_hub import notebook_login
notebook_login()


Then push your model:

trainer.push_to_hub("your-username/persian-bert-emotion-classification")

🔮 Inference Example
from transformers import pipeline
classifier = pipeline("text-classification", model="your-username/persian-bert-emotion-classification")

text = "امروز روز فوق‌العاده‌ای داشتم"
print(classifier(text))


Output:

[{'label': 'positive', 'score': 0.95}]

📌 Next Steps

Try different models (XLM-R, mBERT)

Experiment with longer training & hyperparameters

Collect more Persian data for better performance

Deploy with Gradio or Streamlit

🙌 Acknowledgments

ParsBERT (HooshvareLab)

Hugging Face Transformers

✨ Happy Fine-tuning! ✨
