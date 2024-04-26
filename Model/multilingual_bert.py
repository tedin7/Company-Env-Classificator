import os
# Set CUDA allocator configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from torch.utils.data import random_split

# Constants
DATABASE_URL = "postgresql://tomd:tomd@localhost/classificationenvdb"

# Setup database connection
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
nltk.download('stopwords')

def load_and_preprocess_data():
    """
    Load data and apply basic preprocessing including text cleaning and balancing the dataset.
    """
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

    return balance_dataset(df)

def balance_dataset(data):
    """
    Balance the dataset by resampling the minority class.
    """
    majority = data[data.labels == 0]
    minority = data[data.labels == 1]
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=123)
    return pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

def clean_text(text):
    """
    Clean text data by converting to lowercase, removing punctuation, and filtering out stopwords.
    """
    stop_words = set(stopwords.words('english')).union(set(stopwords.words('spanish')))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def tokenize_and_prepare_data(data):
    """
    Tokenize the data and prepare it for training.
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    def tokenize_function(examples):
        return tokenizer(examples['combined_text'], truncation=True, max_length=512)
    full_dataset = Dataset.from_pandas(data)
    return full_dataset.map(tokenize_function, batched=True), tokenizer, data_collator

def model_init():
    """
    Initialize the DistilBERT model with custom configuration.
    """
    config = DistilBertConfig.from_pretrained(
        'distilbert-base-multilingual-cased',
        num_labels=2,
        id2label={0: "Class0", 1: "Class1"},
        label2id={"Class0": 0, "Class1": 1}
    )
    return DistilBertForSequenceClassification(config)

def compute_metrics(p):
    """
    Compute accuracy, precision, recall, and F1-score of the model predictions.
    """
    pred, labels = p.predictions, p.label_ids
    pred = np.argmax(pred, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='binary', zero_division=0)  # Setting zero_division parameter
    acc = accuracy_score(labels, pred)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def optimize_hyperparameters(full_dataset, tokenizer, data_collator):
    """
    Perform hyperparameter optimization using Optuna without cross-validation.
    """
    # Splitting dataset into train and test sets
    train_size = int(0.7 * len(full_dataset))  # 80% for training
    test_size = len(full_dataset) - train_size  # 20% for testing
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    def objective(trial):
        output_dir = f'./results/trial_{trial.number}'
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=trial.suggest_float('learning_rate', 5e-6, 5e-4),
            per_device_train_batch_size=trial.suggest_categorical('per_device_train_batch_size', [4, 8]),
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            lr_scheduler_type='linear',
            warmup_ratio=0.1,
            logging_steps=500,
            logging_dir=f'{output_dir}/logs',
            gradient_accumulation_steps=4,
            fp16=True,
            use_cpu=False,
            save_total_limit=1,
            do_eval=True
        )
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,  
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        trainer.train()
        eval_result = trainer.evaluate()
        return eval_result["eval_f1"]

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25)

    best_trial = study.best_trial
    best_model_dir = f'./results/trial_{best_trial.number}'

    print(f'Best model directory: {best_model_dir}')

    return study, best_model_dir

def main():
    torch.cuda.empty_cache()
    data = load_and_preprocess_data()
    full_dataset, tokenizer, data_collator = tokenize_and_prepare_data(data)
    study, best_model_dir = optimize_hyperparameters(full_dataset, tokenizer, data_collator)

    print("Best trial:")
    print(f"Value: {study.best_trial.value}")
    print("Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
