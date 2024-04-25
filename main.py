import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, BigInteger, inspect, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy_utils import database_exists, create_database
from datetime import datetime
import json

# Constants
DATA_PATH = 'Data/challenge - dataset.csv'
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://username:password@localhost/databasename")
DB_USER = os.environ.get("DB_USER", "username")

# Setting up the database connection and model
Base = declarative_base()

def initialize_database():
    engine = create_engine(DATABASE_URL)
    if not database_exists(engine.url):
        create_database(engine.url)
    Base.metadata.create_all(engine)
    return engine

engine = initialize_database()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Model definitions
class LinkedInProfile(Base):
    __tablename__ = 'company_profiles'
    id = Column(String, primary_key=True)
    nif_code = Column(Text)
    web_site = Column(Text)
    linkedin_url = Column(Text)
    about = Column(Text)
    label = Column(BigInteger)
    website_url = Column(Text)
    blog_url = Column(Text)
    angellist_url = Column(Text)
    twitter_url = Column(Text)
    facebook_url = Column(Text)
    primary_phone = Column(Text)
    languages = Column(Text)
    alexa_ranking = Column(Integer)
    phone = Column(Text)
    linkedin_uid = Column(Text)
    primary_domain = Column(Text)
    persona_counts = Column(JSON)
    keywords = Column(Text)
    num_suborganizations = Column(Integer)
    short_description = Column(Text)
    specialities_x = Column(Text)
    location_x = Column(Text)
    specialities_y = Column(Text)
    location_y = Column(Text)
    specialities = Column(Text)
    name = Column(Text)
    location = Column(Text)

class APILog(Base):
    __tablename__ = 'api_logs'
    call_id = Column(Integer, primary_key=True)
    request_data = Column(Text)
    response_data = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# Data processing functions
def load_and_preprocess_data():
    data = pd.read_csv(DATA_PATH)
    preprocess_columns = ['about', 'web_site', 'linkedin_url', 'keywords', 'name', 'location']
    for col in preprocess_columns:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)
    return data

def check_and_create_tables(engine):
    # Create an Inspector object
    inspector = inspect(engine)

    # Check if tables exist and create them if they don't
    if not inspector.has_table(LinkedInProfile.__tablename__):
        LinkedInProfile.metadata.create_all(engine)
    if not inspector.has_table(APILog.__tablename__):
        APILog.metadata.create_all(engine)

def validate_and_process_data(data):
    # Drop duplicates based on 'id' column
    data = data.drop_duplicates(subset='id', keep='first')
    
    # Now process each row
    for index, row in data.iterrows():
        # Rename 'Label' column to 'label'
        if 'Label' in row:
            row['label'] = row.pop('Label')
        
        # Validate and process integer fields
        integer_fields = ['alexa_ranking', 'num_suborganizations']
        for field in integer_fields:
            # Check if the value is numeric
            if pd.notna(row[field]):
                try:
                    # Convert the value to an integer
                    int_value = int(row[field])
                    # Check if the value is within the valid range
                    if not (-2147483648 <= int_value <= 2147483647):
                        print(f"Value out of range for field '{field}' at index {index}: {int_value}. Setting to None.")
                        data.at[index, field] = None
                except ValueError:
                    # If the value cannot be converted to an integer, set it to None
                    print(f"Invalid integer value for field '{field}' at index {index}: {row[field]}. Setting to None.")
                    data.at[index, field] = None

        # Convert 'persona_counts' from JSON string to dictionary if it's a string
        if 'persona_counts' in row and isinstance(row['persona_counts'], str):
            try:
                data.at[index, 'persona_counts'] = json.loads(row['persona_counts'])
            except json.JSONDecodeError:
                data.at[index, 'persona_counts'] = {}  # Set default as empty dict if there's an error

    return data


def insert_data_to_db(data):
    session = SessionLocal()
    try:
        for index, row in data.iterrows():
            profile = LinkedInProfile(**row)
            session.add(profile)
        session.commit()
        print("Data inserted successfully into the database.")
    except Exception as e:
        session.rollback()
        print(f"An error occurred: {e}")
    finally:
        session.close()

def main():
    # Check and create tables if they don't exist
    check_and_create_tables(engine)
    
    # Load data
    data_df = load_and_preprocess_data()
    
    # Validate and process data
    data_df = validate_and_process_data(data_df)
    
    # Insert data into the database
    insert_data_to_db(data_df)

# Main execution
if __name__ == "__main__":
    main()
