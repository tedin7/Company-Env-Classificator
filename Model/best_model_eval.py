import os
import pandas as pd
import torch

torch.cuda.empty_cache()
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from nltk.corpus import stopwords
from sklearn.utils import resample
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
import string
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from datasets import Dataset
import re
# Download NLTK stopwords
nltk.download('stopwords')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setup
DB_USER = os.environ.get("DB_USER", "username")  # Use the DB_USER variable
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://username:password@localhost/databasename")
print("Connecting to database URL:", DATABASE_URL)
BEST_MODEL_DIRECTORY = 'best_model/'
plot_directory = 'Evaluation/Plot'
os.makedirs(plot_directory, exist_ok=True)

# Database connection
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Set up the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model = DistilBertForSequenceClassification.from_pretrained(BEST_MODEL_DIRECTORY)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def load_and_preprocess_data():
    session = Session()
    try:
        sql_query = "SELECT about, keywords, label FROM company_profiles"
        df = pd.read_sql_query(sql_query, con=engine)
        stop_words = set(stopwords.words('english')).union(stopwords.words('spanish'))


        def clean_text(text):
            # Remove URLs more generally
            text = re.sub(r'http\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove all http and https URLs
            # Remove strings starting with "www" (e.g., "wwwexamplecom")
            text = re.sub(r'\bwww\.\S+', '', text, flags=re.MULTILINE)  # This targets strings that start with www.
            # Remove punctuation and lower case the text
            text = text.lower().translate(str.maketrans('', '', string.punctuation))
            # Remove stop words
            text = ' '.join([word for word in text.split() if word not in stop_words])
            return text

        df['combined_text'] = df.apply(lambda x: clean_text(x['about'] + " " + x['keywords']), axis=1)
        df.rename(columns={'label': 'labels'}, inplace=True)
        df.dropna(subset=['combined_text', 'labels'], inplace=True)
    finally:
        session.close()
    return balance_dataset(df)

def balance_dataset(data):
    balanced_data = pd.concat([
        resample(data[data.labels == 0], replace=True, n_samples=len(data[data.labels == 1]), random_state=42),
        data[data.labels == 1]
    ]).sample(frac=1, random_state=42)
    return balanced_data


def tokenize_and_prepare_data(data):
    # Ensure tokenization outputs tensors
    tokenized_inputs = tokenizer(data['combined_text'].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")

    # Convert all data directly to tensors and ensure they're sent to the correct device
    input_ids = tokenized_inputs['input_ids'].to(device)
    attention_mask = tokenized_inputs['attention_mask'].to(device)
    labels = torch.tensor(data['labels'].tolist()).to(device)

    # Prepare a custom dataset
    class CustomDataset(Dataset):
        def __init__(self, input_ids, attention_mask, labels):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx]
            }
            return item

    dataset = CustomDataset(input_ids, attention_mask, labels)
    return DataLoader(dataset, batch_size=1, shuffle=True)


def plot_confusion_matrix(true_labels, pred_labels, save_path='Evaluation/Plot/confusion_matrix.png'):
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(true_labels, pred_scores, save_path='Evaluation/Plot/roc_curve.png'):
    fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.title('ROC Curve')
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(true_labels, pred_scores, save_path='Evaluation/Plot/precision_recall_curve.png'):
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot()
    plt.title('Precision-Recall Curve')
    plt.savefig(save_path)
    plt.close()

def interpret_model(data_loader):
    model.eval()  # Ensure the model is in evaluation mode
    true_labels, pred_labels, pred_scores = [], [], []

    for batch in tqdm(data_loader, desc="Evaluating"):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=-1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())
        pred_scores.extend(probs[:, 1].cpu().numpy())

    print(classification_report(true_labels, pred_labels))

    
    # Generate and save plots
    plot_confusion_matrix(true_labels, pred_labels)
    plot_roc_curve(true_labels, pred_scores)
    plot_precision_recall_curve(true_labels, pred_scores)


def get_predict_proba_function():
    def predict_proba(texts):
        model.eval()
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        return probabilities
    return predict_proba


def generate_lime_explanations(df_sample, predict_proba):
    explainer = LimeTextExplainer(class_names=['Not Sustainable', 'Sustainable'])
    explanations = []
    labels = []  # List to store labels
    for _, row in df_sample.iterrows():
        torch.cuda.empty_cache()
        text_instance = row['combined_text']
        original_label = row['labels']  # Assuming the label column is named 'labels'
        exp = explainer.explain_instance(text_instance, predict_proba, num_features=10, num_samples=20)
        explanations.append(exp)
        labels.append(original_label)  # Store the label
    return explanations, labels

def save_explanations_to_html(explanations, labels, filename='Evaluation/LIME Explanations/lime_explanations.html'):
    label_1_explanations = []
    label_0_explanations = []

    # First, collect significant explanations for each label
    for exp, label in zip(explanations, labels):
        features = exp.as_list()
        significant_features = [feature for feature in features if abs(feature[1]) > 0.001]
        if significant_features:
            if label == 1:
                label_1_explanations.append((exp, label))
            else:
                label_0_explanations.append((exp, label))

    # Ensure exactly 5 significant explanations for each label if available
    selected_explanations = label_0_explanations[:2] + label_1_explanations[:2]

    # Write to HTML
    with open(filename, 'w') as f:
        f.write("<html><head><title>LIME Explanations</title><style>body { font-family: Arial; } h1, h2 { color: navy; } .explanation { margin-bottom: 20px; }</style></head><body>")
        for exp, label in selected_explanations:
            f.write(f"<div class='explanation'><h2>Explanation for text with Original Label: {label}</h2>")
            exp_html = exp.as_html()
            f.write(exp_html + "</div>")
        f.write("</body></html>")




def main():
    data = load_and_preprocess_data()
    data_loader = tokenize_and_prepare_data(data)
    #interpret_model(data_loader)
    torch.cuda.empty_cache()
    predict_proba = get_predict_proba_function()
    df_sample = data.sample(n=200, random_state=42) 
    explanations, labels = generate_lime_explanations(df_sample, predict_proba)  # Get explanations and labels
    save_explanations_to_html(explanations, labels)  # Pass both to the HTML function

if __name__ == "__main__":
    main()
