import torch
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.utils import resample
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import shap
import os
import numpy as np
nltk.download('stopwords')
DATABASE_URL = "postgresql://tomd:tomd@localhost/ClassificationEnvDB1"
BEST_MODEL_DIRECTORY = '/home/tomd/Documents/GitHub/Company-Env-Classificator/best_model'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup database connection
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
import os

# Ensure the 'Plot' directory exists
plot_directory = 'Evaluation/Plot'
os.makedirs(plot_directory, exist_ok=True)
def load_and_preprocess_data():
    session = Session()
    try:
        sql_query = "SELECT about, keywords, label FROM company_profiles"
        with engine.connect() as connection:
            df = pd.read_sql_query(sql_query, con=connection)
        df['combined_text'] = df['about'].fillna('') + " " + df['keywords'].fillna('')
        df['combined_text'] = df['combined_text'].apply(clean_text)
        df.rename(columns={'label': 'labels'}, inplace=True)
        df.dropna(subset=['combined_text', 'labels'], inplace=True)
    finally:
        session.close()
    return balance_dataset(df)

def clean_text(text):
    stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join([word for word in text.split() if word not in stop_words])

def balance_dataset(data):
    balanced_data = pd.concat([
        resample(data[data.labels == 1], replace=True, n_samples=len(data[data.labels == 0]), random_state=42),
        data[data.labels == 0]
    ]).sample(frac=1, random_state=42)
    return balanced_data

def tokenize_and_prepare_data(data, tokenizer):
    combined_text = data['combined_text'].tolist()
    tokenized_inputs = tokenizer(combined_text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    tokenized_dataset = Dataset.from_dict({
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': data['labels']
    })
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized_dataset
def plot_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(os.path.join(plot_directory, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(true_labels, pred_scores):
    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.savefig(os.path.join(plot_directory, 'roc_curve.png'))
    plt.close()

def plot_precision_recall_curve(true_labels, pred_scores):
    precision, recall, thresholds = precision_recall_curve(true_labels, pred_scores)
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot()
    plt.savefig(os.path.join(plot_directory, 'precision_recall_curve.png'))
    plt.close()
def interpret_model(model, eval_dataset):
    model.eval()  # Set the model to evaluation mode

    true_labels = []
    pred_labels = []
    pred_scores = []

    with torch.no_grad():  # Deactivate gradients for the following code
        for batch in tqdm(DataLoader(eval_dataset, batch_size=8), desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            pred_scores.extend(outputs.logits.softmax(dim=-1)[:,1].tolist())

            true_labels.extend(labels.tolist())
            pred_labels.extend(preds.tolist())

    print(classification_report(true_labels, pred_labels))
    plot_confusion_matrix(true_labels, pred_labels)
    plot_roc_curve(true_labels, pred_scores)
    plot_precision_recall_curve(true_labels, pred_scores)
def compute_shap_values(model, tokenizer, eval_dataset, device, subset_size=20):
    """
    Compute SHAP values for a subset of the evaluation dataset and save the summary plot to a file.
    
    Parameters:
    model (torch.nn.Module): The trained model.
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer used with the model.
    eval_dataset (torch.utils.data.Dataset): The dataset to compute SHAP values for.
    device (torch.device): The device to perform computation on.
    subset_size (int): The number of samples from the eval_dataset to compute SHAP values for.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Convert the evaluation dataset to a DataLoader
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    
    # Initialize SHAP explainer
    explainer = shap.Explainer(model, tokenizer)

    # We will explain a subset of the data (to save time)
    shap_values_subset = []
    for i, batch in enumerate(eval_loader):
        if i == subset_size:
            break
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # Compute SHAP values
            sv = explainer(input_ids, attention_mask=attention_mask)
            shap_values_subset.append(sv)
    
    # Aggregate the SHAP values
    aggregated_shap_values = np.array([sv.values for sv in shap_values_subset])
    
    # Plot the summary bar plot
    shap.summary_plot(aggregated_shap_values, show=False)
    plt.savefig('Evaluation/Plot/shap_summary_plot.png')
    
    # You can also save the detailed SHAP value plots for each instance
    for i, sv in enumerate(shap_values_subset):
        shap.plots.waterfall(sv[0], show=False)
        plt.savefig(f'shap_waterfall_plot_instance_{i}.png')
        plt.close()
def main():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    model = DistilBertForSequenceClassification.from_pretrained(BEST_MODEL_DIRECTORY)
    model.to(device)
    data = load_and_preprocess_data()
    eval_dataset = tokenize_and_prepare_data(data, tokenizer)
    interpret_model(model, eval_dataset)
    eval_loader = DataLoader(eval_dataset, batch_size=1) 
    compute_shap_values(model, tokenizer, eval_dataset, device, subset_size=20)
    
    
if __name__ == "__main__":
    main()
