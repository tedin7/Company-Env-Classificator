from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker
from database_setup import initialize_database, load_and_preprocess_data, check_and_create_tables, validate_and_process_data, LinkedInProfile, APILog, SessionLocal, insert_profiles_to_db, Base
import uvicorn
import pandas as pd
from API import apimiddleware
from fastapi.responses import JSONResponse
from prediction import predict_text
from typing import Optional, Dict
import numpy as np

# Create FastAPI app instance
app = FastAPI()

# Add custom API middleware
app.add_middleware(apimiddleware.APILoggerMiddleware)

# Set up logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Define Pydantic models for request and response data
class PredictionRequest(BaseModel):
    about: str
    keywords: str

class LinkedInProfileData(BaseModel):
    id: str
    nif_code: Optional[str] = None
    web_site: Optional[str] = None
    linkedin_url: Optional[str] = None
    about: str
    label: Optional[int] = None
    website_url: Optional[str] = None
    blog_url: Optional[str] = None
    angellist_url: Optional[str] = None
    twitter_url: Optional[str] = None
    facebook_url: Optional[str] = None
    primary_phone: Optional[str] = None
    languages: Optional[str] = None
    alexa_ranking: Optional[int] = None
    phone: Optional[str] = None
    linkedin_uid: Optional[int] = None
    primary_domain: Optional[str] = None
    persona_counts: Optional[Dict] = None
    keywords: str
    num_suborganizations: Optional[int] = None
    short_description: Optional[str] = None
    specialities_x: Optional[str] = None
    location_x: Optional[str] = None
    specialities_y: Optional[str] = None
    location_y: Optional[str] = None
    specialities: Optional[str] = None
    name: Optional[str] = None
    location: Optional[str] = None

# Initialize the database and create tables if they do not exist
engine = initialize_database()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(engine)

# Function to initialize the database
def init_db():
    with SessionLocal() as session:
        data = load_and_preprocess_data()
        data = validate_and_process_data(data)
        insert_profiles_to_db(data, session)

# Function to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Event handler for startup
@app.on_event("startup")
async def startup_event():
    init_db()

def convert_numpy_type(value):
    """Convert numpy data types to native Python types."""
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()  # Convert numpy arrays to list if necessary
    return value

# Endpoint to create a LinkedIn profile and perform prediction
@app.post("/predict/")
def create_linkedin_profile(profile_data: LinkedInProfileData, db: Session = Depends(get_db)):
    try:
        # Create new LinkedInProfile instance from the received data
        new_profile = LinkedInProfile(**profile_data.dict())
        db.add(new_profile)
        db.commit()

        # Perform prediction based on `about` and `keywords`
        predicted_label = predict_text(profile_data.about, profile_data.keywords)
        
        # Convert numpy int64 to native int, if necessary
        predicted_label = convert_numpy_type(predicted_label)

        # Update the label after prediction
        new_profile.label = predicted_label
        db.commit()

        return {"predicted_label": predicted_label, "profile_id": new_profile.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to retrieve all LinkedIn profiles
@app.get("/profiles/")
def read_profiles(db: Session = Depends(get_db)):
    try:
        profiles = db.query(LinkedInProfile).all()
        return profiles
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Failed to retrieve profiles", "error": str(e)})

# Endpoint to add a new LinkedIn profile
@app.post("/profiles/")
def add_profile(profile_data: dict, db: Session = Depends(get_db)):
    try:
        data = pd.DataFrame([profile_data])
        validated_data = validate_and_process_data(data)
        insert_profiles_to_db(validated_data, db)
        return {"message": "Profile added successfully", "data": profile_data}
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Failed to add profile", "error": str(e)})

# Endpoint to retrieve API logs
@app.get("/api_logs/")
def read_api_logs(db: Session = Depends(get_db), skip: int = 0, limit: int = 100):
    try:
        logs = db.query(APILog).offset(skip).limit(limit).all()
        return logs
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Failed to retrieve API logs", "error": str(e)})

# Run the FastAPI application using uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)