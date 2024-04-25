# Sustainability Analysis

## Overview

This project implements an end-to-end AI solution for an investment firm focused on identifying sustainability in LinkedIn profiles. The system uses a text classification model to determine if a company's profile is oriented towards sustainability. The project includes robust data management scripts for PostgreSQL database interactions and a FastAPI server for API deployment, ensuring comprehensive data tracking and API usage logging.

## Components


## Data Analysis and Visualization

### Dataset Insights

Our raw dataset includes 2912 entries, each with 28 distinct attributes ranging from basic contact information to detailed textual descriptions. A noteworthy characteristic of this dataset is its bilingual nature: the textual fields contain both English and Spanish, reflecting the diverse linguistic background of the companies analyzed.

### Preprocessing

Given the bilingual text data, we employed a preprocessing pipeline that performs the following steps:
- Convert all text to lowercase to standardize capitalization differences.
- Remove punctuation marks to focus on the textual content.
- Eliminate stopwords from both English and Spanish to reduce noise and concentrate on meaningful words.
- Apply lemmatization to reduce words to their base or dictionary form.

Here's a snippet of Python code that illustrates part of the preprocessing logic:

```python
# Advanced Text Preprocessing
def preprocess_text(text, language='english'):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english')).union(set(stopwords.words('spanish')))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word, pos='n') for word in text.split()])
    return text
```

#### Duplicates Ids

Duplicate rows where removed from the dataset before inserting them in the input table


### Visualization of Findings

#### Class Imbalance

The analysis highlighted a significant class imbalance in the dataset:

![Label Distribution](Exploration/Plot/label_distribution.png)

*Figure 1: The majority of companies in the dataset are not labeled as sustainability-focused.*

#### Text Length Distribution

We also visualized the length of the 'About' and 'Keywords' fields to gauge the verbosity of company descriptions and keyword usage:

![Text Characteristics](Exploration/Plot/text_characteristics.png)

*Figure 2: The 'About' text fields are generally longer than 'Keywords', suggesting more elaborate company descriptions.*

These visualizations are instrumental in understanding the dataset's characteristics and guiding the development of the classification model.


## Model Selection and Training

### Multilingual Model Justification

For this project, i chose the DistilBert multilingual model (`distilbert-base-multilingual-cased`) because of its proven capability to understand multiple languages. This is crucial as our dataset contains a mix of English and Spanish text. A multilingual approach is preferred over translation-based methods for several reasons:

- **Context Preservation**: Translation can sometimes alter the meaning or context of technical and niche terms commonly found in this dataset.

- **Model Efficiency**: Directly using multilingual models allows us to maintain the semantic integrity of the original text, which could be lost through translation errors or inconsistencies.

- **Scalability**: The multilingual model provides scalability. It can handle additional languages without the need for separate translation models or pipelines, which would increase complexity and processing time.

### DistilBert Multilingual Model

We utilize a distilled version of the BERT model which retains most of the original model's performance while being more efficient. The model is fine-tuned on the specific task of classifying text related to sustainability.

Here are the key parameters and configuration details for the model used during training:

- `num_labels`: 2 (Sustainable or Not Sustainable)
- `id2label` and `label2id`: Mapping between the model's numerical labels and human-readable labels
- Optimization is performed using cross-validation with Optuna to find the best hyperparameters such as learning rate and batch size.
- Early stopping is employed to prevent overfitting.

### Training Process

The model underwent a robust training protocol, including:

- **Class imbalance mitigation**: Implemented data balancing to address the class imbalance issue.
- **Hyperparameter optimization**: Utilized Optuna to fine-tune and determine the best model configuration.
- **Trials** : Using 25 trials.

Specifically, the following hyperparameters were optimized:

- `learning_rate`: Tested within a range of 5e-6 to 5e-4 to find the optimal learning pace.
- `per_device_train_batch_size`: Different batch sizes, such as 4 and 8, were experimented with to assess their impact on model performance and training dynamics.
- `num_train_epochs`: The number of complete passes through the entire dataset was tuned for best results.
- `weight_decay`: Included to regularize and reduce the complexity of the model, thus helping prevent overfitting.

In addition, other parameters like the scheduler type and warmup ratio were carefully adjusted to improve training efficiency and ensure proper model convergence.

### Model Interpretation

To ensure the model's trustworthiness and transparency, SHAP values were computed. This step is crucial as it helps interpret the model's predictions, providing insights into how each feature influences the outcome. Such interpretability is vital for validating that the model's decisions are both reasonable and justifiable.

### Implementation Details

The training script makes sophisticated use of a GPU environment, as indicated by setting the CUDA allocator configuration for enhanced efficiency. The DistilBert model is initialized with custom configurations, and the training is facilitated by a sophisticated Trainer object from the Hugging Face's Transformers library, complete with callbacks such as EarlyStoppingCallback to prevent overfitting.

The script effectively manages several tasks including:

- **Data Preprocessing**: Incorporating cleaning and balancing of the dataset.
- **Tokenization**: Leveraging the DistilBert tokenizer for processing the text data.
- **Training**: Executing the model training with hyperparameter optimization through Optuna.
- **Evaluation**: Assessing model performance using accuracy, precision, recall, and F1-score metrics.

## Model Evaluation

### Evaluation Summary

The DistilBert multilingual model was rigorously evaluated on a balanced test set, ensuring an equitable representation of classes. The performance metrics indicate a highly accurate model:

```plaintext
              precision    recall  f1-score   support

           0       0.98      0.99      0.99      2387
           1       0.99      0.98      0.99      2387

    accuracy                           0.99      4774
   macro avg       0.99      0.99      0.99      4774
weighted avg       0.99      0.99      0.99      4774
```

With precision, recall, and F1-scores nearing perfection, the model demonstrates an outstanding ability to classify sustainability orientation in LinkedIn profiles accurately.

### Visual Evaluation Metrics

#### Confusion Matrix

![Confusion Matrix](Evaluation/Plot/confusion_matrix.png)

*Figure 3: Confusion Matrix*

The confusion matrix reveals high true positive and true negative rates, with minimal false positives and negatives, underscoring the model's precision.

#### Precision-Recall Curve

![Precision-Recall Curve](Evaluation/Plot/precision_recall_curve.png)

*Figure 4: Precision-Recall Curve*

The Precision-Recall Curve is tightly bound to the top-right corner, illustrating that the model maintains high precision across all recall levels.

#### ROC Curve

![ROC Curve](Evaluation/Plot/roc_curve.png)

*Figure 5: ROC Curve*

The ROC Curve exhibits an exceptional true positive rate across all false positive rate levels, confirmed by an AUC score of 0.99. This indicates a model with a high discriminative capacity between sustainable and non-sustainable classes.
## Installation

### Prerequisites

- Python 3.6+
- PostgreSQL
- Python packages: SQLAlchemy, FastAPI, uvicorn, pydantic, psycopg2-binary

### Setup Environment

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Configure the Database

1. Install PostgreSQL and create a database.
2. Set the following environment variables based on your operating system:

**For Windows**:

```bash
set DATABASE_URL=postgresql://username:password@localhost/databasename
set DB_USER=username
```

**For Linux and macOS**:

```bash
export DATABASE_URL=postgresql://username:password@localhost/databasename
export DB_USER=username
```
### Initialize the Database

Run the following command to set up your database tables:

```bash
python main.py
```
### Training the model

1. Modify DATABASE_URL in multilingual_bert.py with DB credentials.
2. Run:

```bash
python Model/multilingual_bert.py
```

## Running the API Server

Start the API server by running:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

This command will start the FastAPI server, making the API accessible at `http://localhost:8000`.

## Usage

Send a POST request to `http://localhost:8000/predict/` with a JSON payload containing detailed LinkedIn profile data:

```json
{
  "id": "1",
  "nif_code": "123456",
  "web_site": "example.com",
  "linkedin_url": "linkedin.com/company/example",
  "about": "Company focusing on sustainable energy solutions.",
  "keywords": "sustainability, renewable, green energy",
  "phone": "123-456-7890",
  "location": "City, Country"
}
```

The server will return a prediction indicating whether the profile is focused on sustainability.

## API Endpoints

- **POST `/predict/`**: Receives detailed profile data and returns a classification of "Sustainable" or "Not Sustainable".
- **GET `/data/`**: Provides counts and information about the stored LinkedIn profiles and API calls.

## Data Management

- Use `db_manager.py` to manage profiles and log API activities systematically.

## Troubleshooting

Ensure all environment variables and dependencies are correctly configured. Check the PostgreSQL connection settings if you encounter database errors.
