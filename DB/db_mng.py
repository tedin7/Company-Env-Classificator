# db_manager.py

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

# Update the username, password, and db_name appropriately
DATABASE_URL = "postgresql://username:password@localhost/db_name"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class LinkedInProfile(Base):
    __tablename__ = 'profiles'
    id = Column(Integer, primary_key=True)
    about = Column(Text)
    keywords = Column(Text)
    label = Column(Integer)

class APILog(Base):
    __tablename__ = 'api_logs'
    id = Column(Integer, primary_key=True)
    request_data = Column(Text)
    response_data = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(engine)

def add_profile(about, keywords, label):
    db_session = SessionLocal()
    profile = LinkedInProfile(about=about, keywords=keywords, label=label)
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
