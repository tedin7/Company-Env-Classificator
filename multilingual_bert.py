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

# Constants
DATA_PATH = 'Data/challenge - dataset.csv'

# Download necessary NLTK data
nltk.download('stopwords')

def load_and_preprocess_data():
    """
    Load data and apply basic preprocessing including text cleaning and balancing the dataset.
    """
    data = pd.read_csv(DATA_PATH)
    data['combined_text'] = data['about'].fillna('') + " " + data['keywords'].fillna('')
    data['combined_text'] = data['combined_text'].apply(clean_text)

    if 'Label' in data.columns:
        data.rename(columns={'Label': 'labels'}, inplace=True)

    data = data[['combined_text', 'labels']].dropna()
    return balance_dataset(data)

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
        return tokenizer(examples['combined_text'], truncation=True)

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
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='binary')
    acc = accuracy_score(labels, pred)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def cross_validate(full_dataset, tokenizer, data_collator):
    """
    Perform cross-validation using Optuna for hyperparameter tuning.
    """
    def objective(trial):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = []

        for train_index, test_index in skf.split(np.zeros(len(full_dataset)), full_dataset['labels']):
            train_dataset = full_dataset.select(train_index)
            test_dataset = full_dataset.select(test_index)

            training_args = TrainingArguments(
                output_dir='./results/kfold',
                evaluation_strategy="epoch",
                learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True),
                per_device_train_batch_size=trial.suggest_categorical('per_device_train_batch_size', [4, 8, 16]),
                num_train_epochs=3,
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                use_cpu=False
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
            results.append(eval_result["eval_f1"])

        return np.mean(results)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    return study

def interpret_model_with_shap(model, train_dataset, test_dataset):
    """
    Use SHAP to interpret the model by computing SHAP values for a subset of the test dataset.
    """
    explainer = shap.Explainer(model, train_dataset)
    shap_values = explainer(test_dataset[:100])
    shap.summary_plot(shap_values, feature_names=train_dataset.features)

def main():
    """
    Main function to orchestrate data loading, preprocessing, model training, evaluation, and interpretability analysis.
    """
    data = load_and_preprocess_data()
    full_dataset, tokenizer, data_collator = tokenize_and_prepare_data(data)
    study = cross_validate(full_dataset, tokenizer, data_collator)

    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    model = model_init()
    train_dataset = full_dataset.train_test_split(test_size=0.2)['train']
    test_dataset = full_dataset.train_test_split(test_size=0.2)['test']
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            learning_rate=trial.params['learning_rate'],
            per_device_train_batch_size=trial.params['per_device_train_batch_size']
        ),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()

    interpret_model_with_shap(model, train_dataset, test_dataset)

if __name__ == "__main__":
    main()
