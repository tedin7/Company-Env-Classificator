from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Optional
import uvicorn
from datetime import datetime
import json

app = FastAPI()

class ProfileData(BaseModel):
    id: Optional[int]
    nif_code: Optional[str]
    web_site: Optional[str]
    linkedin_url: Optional[str]
    about: Optional[str]
    label: Optional[int]
    website_url: Optional[str]
    blog_url: Optional[str]
    angellist_url: Optional[str]
    twitter_url: Optional[str]
    facebook_url: Optional[str]
    primary_phone: Optional[str]
    languages: Optional[str]
    alexa_ranking: Optional[int]
    phone: Optional[str]
    linkedin_uid: Optional[str]
    primary_domain: Optional[str]
    persona_counts: Optional[int]
    keywords: Optional[str]
    num_suborganizations: Optional[int]
    short_description: Optional[str]
    specialities_x: Optional[str]
    location_x: Optional[str]
    specialities_y: Optional[str]
    location_y: Optional[str]
    specialities: Optional[str]
    name: Optional[str]
    location: Optional[str]

@app.post("/predict/")
async def predict(data: ProfileData):
    # Convert input data to dict, remove None values
    input_data = {k: v for k, v in data.dict().items() if v is not None}
    # Perform prediction (example logic)
    prediction = 1  # example prediction
    confidence = 0.95  # example confidence

    # Insert API call and result into database
    add_api_log(json.dumps(input_data), json.dumps({"prediction": prediction, "confidence": confidence}))

    # Optionally store all incoming data to company_profiles
    add_profile(input_data)

    return {"prediction": prediction, "confidence": confidence}

@app.get("/data/")
async def get_data_info():
    # Retrieve information about stored data
    db_session = SessionLocal()
    profiles_count = db_session.query(LinkedInProfile).count()
    api_calls_count = db_session.query(APILog).count()
    db_session.close()
    return {"profiles_count": profiles_count, "api_calls_count": api_calls_count}

if __name__ == "__main__":
    init_db()  # Ensure tables are created
    uvicorn.run(app, host="0.0.0.0", port=8000)
