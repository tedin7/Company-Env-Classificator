import polars as pl
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import numpy as np
import random
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords once into memory
stop_words = stopwords.words('english')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[\W_]+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Load data using Polars
df = pl.read_csv('/home/tomd/Documenti/GitHub/Company-Env-Classificator/Data/translated_dataset.csv')

# Fill missing values in 'about' or 'keywords' if any
df = df.with_columns([
    pl.col("about").fill_null("").alias("about"),
    pl.col("keywords").fill_null("").alias("keywords")
])

# Use Polars to apply the preprocessing function
df = df.with_columns([
    pl.col("about").map_elements(preprocess_text, return_dtype=pl.Utf8).alias("processed_about"),
    pl.col("keywords").map_elements(preprocess_text, return_dtype=pl.Utf8).alias("processed_keywords")
])

# Combine 'processed_about' and 'processed_keywords' into a single text field
df = df.with_columns(
    (pl.col("processed_about") + " " + pl.col("processed_keywords")).alias("combined_text")
)

# Convert to pandas DataFrame for sklearn compatibility
df_pd = df.to_pandas()

# Define a Pipeline that includes TF-IDF vectorization, SMOTE, and RandomForestClassifier
pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define a parameter grid: combining TF-IDF parameters with RandomForest parameters
param_grid = {
    'tfidf__max_features': [1000, 2000, 3000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# Configure GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_pd['combined_text'], df_pd['Label'], test_size=0.3, random_state=42)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Predictions and evaluation
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
