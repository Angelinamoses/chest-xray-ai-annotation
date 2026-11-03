
from pydantic import BaseModel
from typing import Dict, List


class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    input_size: List[int]
    output_channels: int
    class_names: List[str]
    thresholds: Dict[str, float]
    device: str


class PredictionResponse(BaseModel):
    filename: str
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    class_names: List[str]
    thresholds: Dict[str, float]
    masks: Dict[str, str]
    probabilities: Dict[str, str]
    device: str
