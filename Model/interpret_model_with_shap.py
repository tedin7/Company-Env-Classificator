import os
import torch
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification,
                          DistilBertConfig, Trainer, TrainingArguments, EarlyStoppingCallback,
                          DataCollatorWithPadding)
from datasets import Dataset
import optuna
import shap
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
import sys
sys.path.append('/home/tomd/Documents/GitHub/Company-Env-Classificator/') 
from main import LinkedInProfile
# Constants
DATABASE_URL = "postgresql://tomd:tomd@localhost/ClassificationEnvDB1"

# Setup database connection
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
nltk.download('stopwords')

def load_and_preprocess_data():
    try:
        sql_query = "SELECT about, keywords, label FROM company_profiles"
         # Execute the query and load into DataFrame
        with engine.connect() as connection:
            df = pd.read_sql_query(sql_query, con=connection)
        # Combine 'about' and 'keywords' into a single text column for processing
        df['combined_text'] = df['about'].fillna('') + " " + df['keywords'].fillna('')
        df['combined_text'] = df['combined_text'].apply(clean_text)

        # Rename 'label' to 'labels' if necessary
        df.rename(columns={'label': 'labels'}, inplace=True)

        # Drop rows with NaN labels if any
        df = df[['combined_text', 'labels']].dropna()
    finally:
        session.close()

    return df

def balance_dataset(data):
    majority = data[data.labels == 0]
    minority = data[data.labels == 1]
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=123)
    return pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

def clean_text(text):
    stop_words = set(stopwords.words('english')).union(set(stopwords.words('spanish')))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def tokenize_and_prepare_data(data):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    def tokenize_function(examples):
        return tokenizer(examples['combined_text'], truncation=True, max_length=512)
    full_dataset = Dataset.from_pandas(data)
    return full_dataset.map(tokenize_function, batched=True), tokenizer, data_collator

def model_init():
    config = DistilBertConfig.from_pretrained(
        'distilbert-base-multilingual-cased',
        num_labels=2,
        id2label={0: "Class0", 1: "Class1"},
        label2id={"Class0": 0, "Class1": 1}
    )
    return DistilBertForSequenceClassification(config)

def compute_metrics(p):
    pred, labels = p.predictions, p.label_ids
    pred = np.argmax(pred, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='binary')
    acc = accuracy_score(labels, pred)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def interpret_model_with_shap(model, train_dataset, test_dataset):
    explainer = shap.Explainer(model, train_dataset)
    shap_values = explainer(test_dataset[:100])
    shap.summary_plot(shap_values, feature_names=train_dataset.features)

def main():
    torch.cuda.empty_cache()
    data = load_and_preprocess_data()
    best_model_directory = f'./results/best_model'
    best_model = DistilBertForSequenceClassification.from_pretrained(best_model_directory, config=model_init().config)

    full_dataset, tokenizer, data_collator = tokenize_and_prepare_data(data)
    # Use the best model for final training and interpretation
    train_dataset = full_dataset.train_test_split(test_size=0.2)['train']
    test_dataset = full_dataset.train_test_split(test_size=0.2)['test']
    trainer = Trainer(
        model=best_model,
        args=TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=best_model.params['learning_rate'],
            per_device_train_batch_size=best_model.params['per_device_train_batch_size'],
            lr_scheduler_type='linear',
            warmup_ratio=0.1,
            logging_dir='./logs',
            fp16=True
        ),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()

    interpret_model_with_shap(best_model, train_dataset, test_dataset)

if __name__ == "__main__":
    main()
