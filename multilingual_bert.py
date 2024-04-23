import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from sklearn.utils import resample
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import optuna
from transformers import DistilBertConfig

# Load and preprocess data
data_path = 'Data/challenge - dataset.csv'
data = pd.read_csv(data_path)
data['combined_text'] = data['about'].fillna('') + " " + data['keywords'].fillna('')

# Check if 'Label' needs to be renamed to 'labels'
if 'Label' in data.columns:
    data.rename(columns={'Label': 'labels'}, inplace=True)

data = data[['combined_text', 'labels']].dropna()

# Resample dataset to balance
majority = data[data.labels == 0]
minority = data[data.labels == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=123)
data_balanced = pd.concat([majority, minority_upsampled])

# Shuffle the data
data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

# Define the tokenization function
def tokenize_function(examples):
    return tokenizer(examples['combined_text'], padding="max_length", truncation=True, max_length=128)

# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(data_balanced)
dataset = dataset.map(tokenize_function, batched=True)
train_test_split = dataset.train_test_split(test_size=0.3)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Define the compute_metrics function
def compute_metrics(p):
    pred, labels = p.predictions, p.label_ids
    pred = np.argmax(pred, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='binary')
    acc = accuracy_score(labels, pred)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Function to instantiate the model
def model_init():
    config = DistilBertConfig.from_pretrained(
        'distilbert-base-multilingual-cased',
        num_labels=2,
        id2label={0: "Class0", 1: "Class1"},
        label2id={"Class0": 0, "Class1": 1}
    )
    model = DistilBertForSequenceClassification(config)
    return model

# Optuna objective function
def objective(trial):
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True),
        per_device_train_batch_size=trial.suggest_categorical('per_device_train_batch_size', [4, 8, 16]),
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
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
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result["eval_f1"]

# Run the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Output the best trial results
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
