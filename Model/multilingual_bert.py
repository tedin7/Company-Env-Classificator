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

def cross_validate(full_dataset, tokenizer, data_collator):
    def objective(trial):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []
        for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(full_dataset)), full_dataset['labels'])):

            train_dataset = full_dataset.select(train_index)
            test_dataset = full_dataset.select(test_index)
            torch.cuda.empty_cache()
            training_args = TrainingArguments(
                output_dir=f'./results/kfold/trial_{trial.number}_fold_{fold}',
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=trial.suggest_float('learning_rate', 5e-6, 5e-4),
                per_device_train_batch_size=trial.suggest_categorical('per_device_train_batch_size', [4, 8]),
                num_train_epochs=3,
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                lr_scheduler_type='linear',
                warmup_ratio=0.1,
                logging_steps=500,
                gradient_accumulation_steps=4,
                fp16=True,
                use_cpu=False,
                save_total_limit=1,
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
            trainer.save_model(f'./results/best_model')
            eval_result = trainer.evaluate()
            fold_results.append((eval_result["eval_f1"], fold))

        # Select the best fold
        best_f1, best_fold = max(fold_results, key=lambda x: x[0])
        trial.set_user_attr('best_fold', best_fold)  # Save the best fold in the trial's user attributes

        return best_f1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    return study
def main():
    torch.cuda.empty_cache()
    data = load_and_preprocess_data()
    full_dataset, tokenizer, data_collator = tokenize_and_prepare_data(data)
    study = cross_validate(full_dataset, tokenizer, data_collator)

    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
