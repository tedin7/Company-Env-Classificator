import os
import json
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, BigInteger, inspect, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy_utils import database_exists, create_database
from datetime import datetime

# Constants
DATA_PATH = 'Data/challenge - dataset.csv'
DB_USER = os.environ.get("DB_USER", "username")  # Use the DB_USER variable
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://username:password@localhost/databasename")
print("Connecting to database URL:", DATABASE_URL)

# Setting up the database connection and model
Base = declarative_base()

def initialize_database():
    """Initialize the database engine and create tables if they do not exist."""
    engine = create_engine(DATABASE_URL)
    if not database_exists(engine.url):
        create_database(engine.url)
    Base.metadata.create_all(engine)
    return engine

engine = initialize_database()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Model definitions
class LinkedInProfile(Base):
    """Define the LinkedInProfile table model."""
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
    alexa_ranking = Column(BigInteger)
    phone = Column(Text)
    linkedin_uid = Column(BigInteger)
    primary_domain = Column(Text)
    persona_counts = Column(JSON)
    keywords = Column(Text)
    num_suborganizations = Column(BigInteger)
    short_description = Column(Text)
    specialities_x = Column(Text)
    location_x = Column(Text)
    specialities_y = Column(Text)
    location_y = Column(Text)
    specialities = Column(Text)
    name = Column(Text)
    location = Column(Text)

class APILog(Base):
    """Define the APILog table model."""
    __tablename__ = 'api_logs'
    call_id = Column(Integer, primary_key=True)
    request_data = Column(Text)
    response_data = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# Data processing functions
def load_and_preprocess_data():
    """Load and preprocess data from CSV."""
    data = pd.read_csv(DATA_PATH)
    preprocess_columns = ['about', 'web_site', 'linkedin_url', 'keywords', 'name', 'location']
    for col in preprocess_columns:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)
    return data

def check_and_create_tables(engine):
    """Check if tables exist and create them if necessary."""
    inspector = inspect(engine)
    if not inspector.has_table(LinkedInProfile.__tablename__):
        LinkedInProfile.metadata.create_all(engine)
    if not inspector.has_table(APILog.__tablename__):
        APILog.metadata.create_all(engine)

def validate_and_process_data(data):
    """Validate and process data."""
    data = data.drop_duplicates(subset='id', keep='first')
    for column in data.columns:
        if data[column].dtype == float:
            data.loc[:, column] = data[column].fillna(0)
        elif data[column].dtype == object:
            data.loc[:, column] = data[column].fillna('Unknown')
        else:
            data.loc[:, column] = data[column].where(pd.notna(data[column]), None)

    for index, row in data.iterrows():
        if 'Label' in row:
            data.at[index, 'label'] = row['Label']
        for column in data.columns:
            if data[column].dtype in [int, float]:
                try:
                    int_value = int(row[column])
                    if not (-2147483647 <= int_value <= 2147483647):
                        data.at[index, column] = None
                except ValueError:
                    data.at[index, column] = None
            if column == 'persona_counts' and isinstance(row[column], str):
                try:
                    data.at[index, column] = json.loads(row[column])
                except json.JSONDecodeError:
                    data.at[index, column] = {}
    return data

def insert_profiles_to_db(data, session):
    """Insert profiles data into the database."""
    try:
        for index, row in data.iterrows():
            profile = LinkedInProfile(**row.to_dict())
            session.add(profile)
        session.commit()
        print("Data inserted successfully into the database.")
    except Exception as e:
        session.rollback()
        print(f"Data already Inserted")

def init_db():
    """Initialize the database with data."""
    engine = initialize_database()
    SessionLocal.configure(bind=engine)
    session = SessionLocal()
    data = load_and_preprocess_data()
    data = validate_and_process_data(data)
    insert_profiles_to_db(data, session)
    session.close()

if __name__ == "__main__":
    main()