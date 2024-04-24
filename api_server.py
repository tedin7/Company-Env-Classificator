# api_server.py

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from db_manager import add_api_log, LinkedInProfile, SessionLocal  # Import from your DB manager script

app = FastAPI()

class Item(BaseModel):
    text: str

@app.post("/predict/")
async def create_item(item: Item):
    # Example: Process the text and predict
    # This is where you'd clean the text and use your model to predict
    prediction = "Sustainable" if "sustainable" in item.text.lower() else "Not Sustainable"

    # Log API request and response
    add_api_log(request_data=item.text, response_data=prediction)

    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
