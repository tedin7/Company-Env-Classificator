# Classification on Companies Sustainability

## Overview

This project implements an end-to-end AI solution for an investment firm focused on identifying sustainability in LinkedIn profiles. The system uses a text classification model to determine if a company's profile is oriented towards sustainability. The project includes robust data management scripts for PostgreSQL database interactions and a FastAPI server for API deployment, ensuring comprehensive data tracking and API usage logging.


# Data Analysis and Visualization

### Dataset Insights

Our raw dataset includes 2912 entries, each with 28 distinct attributes ranging from basic contact information to detailed textual descriptions. A noteworthy characteristic of this dataset is its bilingual nature: the textual fields contain both English and Spanish, reflecting the diverse linguistic background of the companies analyzed.

### Duplicates Ids

Some Duplicate rows where present in the original dataset and have benn removed from the dataset before inserting them in the input table i used to train and test the model.

### Preprocessing

Given the bilingual text data, i employed a preprocessing pipeline that performs the following steps:
- Convert all text to lowercase to standardize capitalization differences.
- Remove punctuation marks to focus on the textual content.
- Eliminate stopwords from both English and Spanish to reduce noise and concentrate on meaningful words.
- Apply lemmatization to reduce words to their base or dictionary form.


## Visualization and Findings

### Class Imbalance

The analysis highlighted a significant class imbalance in the dataset:

![Label Distribution](Exploration/Plot/label_distribution.png)

*Figure 1: The majority of companies in the dataset are not labeled as sustainability-focused.*

### Text Length Distribution

I also visualized the length of the 'About' and 'Keywords' fields to gauge the verbosity of company descriptions and keyword usage:

![Text Characteristics](Exploration/Plot/text_characteristics.png)

*Figure 2: The 'About' text fields are generally longer than 'Keywords', suggesting more elaborate company descriptions.*

These visualizations are instrumental in understanding the dataset's characteristics and guiding the development of the classification model.


## Model Selection and Training

### Multilingual Model

For this project, i chose the DistilBert multilingual model (`distilbert-base-multilingual-cased`) because of its proven capability to understand multiple languages. This is crucial as our dataset contains a mix of English and Spanish text. A multilingual approach is preferred over translation-based methods for several reasons:

- **Context Preservation**: Translation can sometimes alter the meaning or context of technical and niche terms commonly found in this dataset.

- **Model Efficiency**: Directly using multilingual models allows us to maintain the semantic integrity of the original text, which could be lost through translation errors or inconsistencies.

- **Scalability**: The multilingual model provides scalability. It can handle additional languages without the need for separate translation models or pipelines, which would increase complexity and processing time.

### DistilBert Multilingual Model

I utilize a distilled version of the BERT model which retains most of the original model's performance while being more efficient. The model is fine-tuned on the specific task of classifying text related to sustainability.

Here are the key parameters and configuration details for the model used during training:

- `num_labels`: 2 (Sustainable or Not Sustainable)
- `id2label` and `label2id`: Mapping between the model's numerical labels and human-readable labels
- Optimization is performed using Optuna to find the best hyperparameters such as learning rate and batch size.
- Early stopping is employed to prevent overfitting.

### Training Process

The model underwent a robust training, including:

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

To ensure the model's trustworthiness and transparency, i used LIME. This step is crucial as it helps interpret the model's predictions, providing insights into how each feature influences the outcome. Such interpretability is vital for validating that the model's decisions are both reasonable and justifiable.

### Implementation Details

The training script makes use of a GPU environment(Nvidia 2060 GPU), as indicated by setting the CUDA allocator configuration for enhanced efficiency. The DistilBert model is initialized with custom configurations, and the training is facilitated by a Trainer object from the Hugging Face's Transformers library, complete with callbacks such as EarlyStoppingCallback to prevent overfitting.

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

With precision, recall, and F1-scores of 0.99, the model demonstrates an outstanding ability to classify the sustainability of a company from LinkedIn data accurately.

## Visual Evaluation Metrics

### Confusion Matrix

![Confusion Matrix](Evaluation/Plot/confusion_matrix.png)

*Figure 3: Confusion Matrix*

The confusion matrix reveals high true positive and true negative rates, with minimal false positives and negatives, underscoring the model's precision.

### Precision-Recall Curve

![Precision-Recall Curve](Evaluation/Plot/precision_recall_curve.png)

*Figure 4: Precision-Recall Curve*

The Precision-Recall Curve is tightly bound to the top-right corner, illustrating that the model maintains high precision across all recall levels.

### ROC Curve

![ROC Curve](Evaluation/Plot/roc_curve.png)

*Figure 5: ROC Curve*

The ROC Curve exhibits an exceptional true positive rate across all false positive rate levels, confirmed by an AUC score of 0.99. This indicates a model with a high discriminative capacity between sustainable and non-sustainable classes.

### Lime Explanations

We used Lime explanations to interpret the predictions of our model. Below are explanations for some sample records:

![LIME Explanation 1](Evaluation/LIME%20Explanations/output_image_combined.png)

# Installation

### Prerequisites

- Python 3.6+
- PostgreSQL
- Python packages: torch, pandas, numpy, nltk, scikit-learn, transformers, datasets, optuna, shap, seaborn, fastapi, uvicorn, pydantic, sqlalchemy, psycopg2-binary, sqlalchemy_utils, accelerate, lime

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
### Initialize the API Server and Database

Run the following command to set up your database tables and API Server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

This command will start the FastAPI server,DB, making the API accessible at `http://localhost:8000`.


---

# API Endpoints Documentation

### POST `/predict/`
Receives detailed LinkedIn profile data, performs a prediction based on certain fields, and updates the profile with the predicted label.

#### Usage
**Endpoint**: `http://localhost:8000/predict/`

**Payload**:
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

**Response**:
```json
{
  "predicted_label": 1,
  "profile_id": "1"
}
```
This endpoint updates a LinkedIn profile with a prediction on its focus (e.g., sustainability) based on the 'about' and 'keywords' fields.

---

### GET `/profiles/`
Fetches all LinkedIn profiles from the database.

#### Usage
**Endpoint**: `http://localhost:8000/profiles/`

**Response**:
```json
[
  {
    "id": "1",
    "about": "Details about the company...",
    "keywords": "sustainability, renewable"
    // Additional profile fields
  },
  // More profiles
]
```
Returns a list of all LinkedIn profiles stored in the database.

---

### POST `/profiles/`
Adds a new LinkedIn profile to the database after validating and preprocessing the data.

#### Usage
**Endpoint**: `http://localhost:8000/profiles/`

**Payload**:
```json
{
  "id": "2",
  "about": "Innovative tech company.",
  "keywords": "innovation, technology",
  "nif_code": "654321",
  "web_site": "tech-example.com"
}
```

**Response**:
```json
{
  "message": "Profile added successfully",
  "data": {
    "id": "2",
    "about": "Innovative tech company.",
    "keywords": "innovation, technology",
    "nif_code": "654321",
    "web_site": "tech-example.com"
  }
}
```
Adds a new profile to the database, ensuring the data is processed and stored correctly.

---

### GET `/api_logs/`
Retrieves logs of API usage, which include details on the requests made and responses sent.

#### Usage
**Endpoint**: `http://localhost:8000/api_logs/`

**Query Parameters**:
- `skip` (optional): Number of entries to skip for pagination.
- `limit` (optional): Maximum number of log entries to return.

**Response**:
```json
[
  {
    "request_data": "{...}",
    "response_data": "{...}",
    // Additional log details
  },
  // More logs
]
```
Provides a way to monitor and review the API usage by returning logs of the past API calls.

---


## Training the model

1. Modify DATABASE_URL in multilingual_bert.py with DB credentials.
2. Run:

```bash
python Model/multilingual_bert.py
```

## Using Pre-trained Model

- If you want to use the already trained model, you can find it in the email as "best_model.zip". Extract the contents of the zip file to access the pre-trained model.

## Data Management

- Use `database_setup.py` and `apimiddleware.py` to manage profiles and log API activities systematically.

## Troubleshooting

Ensure all environment variables and dependencies are correctly configured. Check the PostgreSQL connection settings if you encounter database errors.
