from pydantic import BaseModel, Field
from typing import List, Dict

class PredictionRequest(BaseModel):
    data: Dict
    category: List

class PredictionResponse(BaseModel):
    predictions: List

    

