from pydantic import BaseModel, Field
from typing import List, Dict

class PredictionRequest(BaseModel):
    data: Dict
    category: str

class PredictionResponse(BaseModel):
    predictions: List

    

