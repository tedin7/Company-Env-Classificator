from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer

app = FastAPI()

# Load your pre-trained model (as before)
model = model_init()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

class RequestBody(BaseModel):
    about: str
    keywords: str

@app.post('/predict')
async def predict(request: RequestBody):
    text = clean_text(request.about + " " + request.keywords)
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_prob = torch.softmax(logits, dim=-1)
    prediction = pred_prob.argmax().item()
    confidence = pred_prob[0, prediction].item()

    # Store the API call and prediction in the database
    insert_api_call(request.about + " " + request.keywords, prediction, confidence)

    return {'prediction': prediction, 'confidence': confidence}

if __name__ == '__main__':
    create_tables()  # Ensure tables are created
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
