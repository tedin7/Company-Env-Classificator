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
2. Modify `DATABASE_URL` in `db_manager.py` with your database credentials.

### Initialize the Database

Run the following command to set up your database tables:

```bash
python db_manager.py
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
