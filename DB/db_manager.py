from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

# Update the username, password, and db_name appropriately
DATABASE_URL = "postgresql://username:password@localhost/db_name"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class LinkedInProfile(Base):
    __tablename__ = 'company_profiles'
    company_id = Column(Integer, primary_key=True)
    id = Column(BigInteger)
    nif_code = Column(String)
    web_site = Column(String)
    linkedin_url = Column(String)
    about = Column(Text)
    label = Column(Integer)
    website_url = Column(String)
    blog_url = Column(String)
    angellist_url = Column(String)
    twitter_url = Column(String)
    facebook_url = Column(String)
    primary_phone = Column(String)
    languages = Column(String)
    alexa_ranking = Column(Integer)
    phone = Column(String)
    linkedin_uid = Column(String)
    primary_domain = Column(String)
    persona_counts = Column(Integer)
    keywords = Column(Text)
    num_suborganizations = Column(Integer)
    short_description = Column(Text)
    specialities_x = Column(String)
    location_x = Column(String)
    specialities_y = Column(String)
    location_y = Column(String)
    specialities = Column(String)
    name = Column(String)
    location = Column(String)

class APILog(Base):
    __tablename__ = 'api_logs'
    call_id = Column(Integer, primary_key=True)
    request_data = Column(Text)
    response_data = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(engine)

def add_profile(data_dict):
    db_session = SessionLocal()
    profile = LinkedInProfile(**data_dict)
    db_session.add(profile)
    db_session.commit()
    db_session.close()

def add_api_log(request_data, response_data):
    db_session = SessionLocal()
    log = APILog(request_data=str(request_data), response_data=str(response_data))
    db_session.add(log)
    db_session.commit()
    db_session.close()

if __name__ == "__main__":
    init_db()
    print("Database and tables are set up.")
