import numpy as np
import pandas as pd
import polars as pl
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[\W_]+', ' ', text)
    tokens = word_tokenize(text)
    stop_words = stopwords.words('english')
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Load data using Polars
df = pl.read_csv('/home/tomd/Documenti/GitHub/Company-Env-Classificator/Data/translated_dataset.csv').to_pandas()

# Combine 'about' and 'keywords', and apply preprocessing
df['about'] = df['about'].fillna("")
df['keywords'] = df['keywords'].fillna("")

df['combined_text'] = df['about'] + " " + df['keywords']
df['combined_text'] = df['combined_text'].apply(preprocess_text)

# Feature extraction setup
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Define the model components
logistic_model = LogisticRegression(class_weight='balanced')
random_forest_model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Ensemble model using VotingClassifier
ensemble_model = VotingClassifier(estimators=[
    ('logistic', logistic_model),
    ('random_forest', random_forest_model)
], voting='soft')  # Use 'soft' if you want to weigh probabilities

# Setup the pipeline with vectorization, SMOTE, and the ensemble classifier
pipeline = ImbPipeline([
    ('tfidf', vectorizer),
    ('smote', SMOTE(random_state=42)),
    ('classifier', ensemble_model)
])

# Define parameter grid for grid search
param_grid = {
    'tfidf__max_features': [1000, 2000, 3000, 4000, 5000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'classifier__logistic__C': [0.01, 0.1, 1, 10],
    'classifier__logistic__penalty': ['l2'],
    'classifier__random_forest__n_estimators': [100, 200],
    'classifier__random_forest__max_depth': [None, 10, 20],
    'classifier__random_forest__min_samples_split': [2, 5],
    'classifier__random_forest__min_samples_leaf': [1, 2]
}

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['combined_text'], df['Label'], test_size=0.3, random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model after grid search
best_model = grid_search.best_estimator_

# Predictions and evaluation
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
