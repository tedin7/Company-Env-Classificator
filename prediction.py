from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import nltk
from nltk.corpus import stopwords
import string

# Function to predict text classification using DistilBERT model
def predict_text(about, keywords):
    # Load the trained model
    best_model_dir = 'best_model/'
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    model = DistilBertForSequenceClassification.from_pretrained(best_model_dir)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # Download NLTK stopwords if not already downloaded
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english')).union(stopwords.words('spanish'))  # Get English and Spanish stopwords
    punctuation_trans = str.maketrans('', '', string.punctuation)  # Translation table for removing punctuation

    # Function to preprocess text data
    def preprocess_text(text):
        return ' '.join([word for word in text.lower().translate(punctuation_trans).split() if word not in stop_words])

    combined_text = f"{about} {keywords}"  # Combine about and keywords text
    processed_text = preprocess_text(combined_text)  # Preprocess the combined text
    tokenized_input = tokenizer(processed_text, truncation=True, padding=True, max_length=512, return_tensors="pt")  # Tokenize input text

    if torch.cuda.is_available():
        tokenized_input = {key: value.cuda() for key, value in tokenized_input.items()}  # Move tokenized input to GPU if available

    with torch.no_grad():
        output = model(**tokenized_input)  # Forward pass through the model
        probabilities = torch.nn.functional.softmax(output.logits, dim=1).cpu().numpy()  # Calculate softmax probabilities
        predicted_label = probabilities.argmax()  # Get the predicted label index

    return predicted_label  # Return the predicted label
