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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Ensure nltk resources are downloaded
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
df = pl.read_csv('Data/translated_dataset.csv')

# Check for missing values and fill them
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
df_pd['combined_text'] = df_pd['processed_about'] + " " + df_pd['processed_keywords']

# Define a Pipeline that includes TF-IDF vectorization, SMOTE, and Logistic Regression
pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(random_state=42))
])

# Define a parameter grid: combining TF-IDF parameters with logistic regression parameters
param_grid = {
    'tfidf__max_features': [1000, 2000, 3000,4000,5000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l2']
}

# Configure GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(df_pd['combined_text'], df_pd['Label'])

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Extract best Logistic Regression model and vectorizer for feature importance analysis
best_clf = best_model.named_steps['clf']
best_tfidf = best_model.named_steps['tfidf']

# Feature Importance
def plot_feature_importance(model, vectorizer, filename='feature_importance.png'):
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_.flatten()
    sorted_indices = np.argsort(coefs)
    
    top_positive_indices = sorted_indices[-10:]  # top 10 positive
    top_negative_indices = sorted_indices[:10]  # top 10 negative

    top_coefficients = np.hstack([top_negative_indices, top_positive_indices])
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coefs[top_coefficients]]
    plt.bar(np.arange(2 * 10), coefs[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * 10), feature_names[top_coefficients], rotation=60, ha='right')
    plt.title('Feature Importance')
    plt.ylabel('Coefficient Magnitude')
    plt.xlabel('Features')

    plt.savefig(filename)  # Save the plot to a file
    plt.close()  # Close the plot to free up memory

# Example usage:
plot_feature_importance(best_clf, best_tfidf, 'my_feature_importance.png')


# Predictions and evaluation
X_train, X_test, y_train, y_test = train_test_split(df_pd['combined_text'], df_pd['Label'], test_size=0.3, random_state=42)
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
