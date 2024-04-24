# Sustainability Analysis API

## Overview

This project implements an end-to-end AI solution for an investment firm focused on identifying sustainability in LinkedIn profiles. The system uses a text classification model to determine if a company's profile is oriented towards sustainability. The project includes data management scripts for PostgreSQL database interactions and a FastAPI server for API deployment.

## Components

1. **Database Management Script (`db_manager.py`)**:
   - Handles the creation and management of a PostgreSQL database.
   - Sets up tables for storing LinkedIn profile data and API logs.

2. **API Server Script (`api_server.py`)**:
   - Serves the predictive model via a FastAPI server.
   - Logs each API request and its response for auditing and traceability.

## Installation

### Prerequisites

- Python 3.6+
- PostgreSQL
- Python packages: SQLAlchemy, FastAPI, uvicorn, pydantic, psycopg2-binary

### Setup Environment

Install the required Python packages:

```bash
pip install fastapi uvicorn pydantic sqlalchemy psycopg2-binary
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

Send a POST request to `http://localhost:8000/predict/` with a JSON payload containing the LinkedIn text data:

```json
{
  "text": "Example text from a LinkedIn profile discussing sustainability."
}
```

The server will return a prediction indicating whether the text suggests that the profile is focused on sustainability.

## API Endpoints

- **POST `/predict/`**: Receives text data and returns a classification of "Sustainable" or "Not Sustainable".

## Data Management

- Use `db_manager.py` to add profiles to the database or to log API activities.

## Troubleshooting

Ensure all environment variables and dependencies are correctly configured. Check the PostgreSQL connection settings if you encounter database errors.

