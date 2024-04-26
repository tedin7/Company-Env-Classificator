import pandas as pd
import time
from sqlalchemy import create_engine
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import nltk
from nltk.corpus import stopwords
import string
from sqlalchemy.orm import sessionmaker

# Define the database URL
DATABASE_URL = "postgresql://tomd:tomd@localhost/classificationenvdb"

# Setup database connection
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Retrieve data from the database
sql_query = "SELECT about, keywords, label FROM company_profiles LIMIT 2500"
with engine.connect() as connection:
    df_new_data = pd.read_sql_query(sql_query, con=connection)

# Load the trained model
best_model_dir = '/home/tomd/Documents/GitHub/Company-Env-Classificator/best_model'
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model = DistilBertForSequenceClassification.from_pretrained(best_model_dir)
model.eval()  # Set the model to evaluation mode
if torch.cuda.is_available():
    model.cuda()  # Move the model to GPU

nltk.download('stopwords')
stop_words = set(stopwords.words('english')).union(stopwords.words('spanish'))
punctuation_trans = str.maketrans('', '', string.punctuation)

def preprocess_text(text):
    return ' '.join([word for word in text.lower().translate(punctuation_trans).split() if word not in stop_words])

def predict_new_data(new_data):
    start_time = time.time()  # Start timing
    predictions = []
    
    for _, row in new_data.iterrows():
        combined_text = f"{row['about'] or ''} {row['keywords'] or ''}"
        processed_text = preprocess_text(combined_text)
        tokenized_input = tokenizer(processed_text, truncation=True, padding=True, max_length=512, return_tensors="pt")
        
        if torch.cuda.is_available():
            tokenized_input = {key: value.cuda() for key, value in tokenized_input.items()}  # Move tensor to GPU
        
        with torch.no_grad():
            output = model(**tokenized_input)
            probabilities = torch.nn.functional.softmax(output.logits, dim=1).cpu().numpy()
            predicted_label = probabilities.argmax()
            predictions.append(predicted_label)

    total_time = time.time() - start_time
    time_per_prediction = total_time / len(new_data)

    return predictions, time_per_prediction

predicted_labels, time_per_prediction = predict_new_data(df_new_data)
correct_predictions = sum([pred == actual for pred, actual in zip(predicted_labels, df_new_data['label'])])
classification_rate = correct_predictions / len(df_new_data)

print("Predicted class names:", predicted_labels)
print("Original labels:", df_new_data['label'].tolist())
print("Time per prediction:", time_per_prediction)
print("Classification rate:", classification_rate)
