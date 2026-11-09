
from pydantic import BaseModel
from typing import Dict, List, Optional


# --------------------------------------------------
# MODEL INFORMATION RESPONSE
# --------------------------------------------------

class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    input_size: List[int]
    output_channels: int
    class_names: List[str]
    thresholds: Dict[str, float]
    device: str


# --------------------------------------------------
# PREDICTION RESPONSE
# --------------------------------------------------

class PredictionResponse(BaseModel):

    # File and model information
    filename: str
    model_version: str

    # Image dimensions
    original_width: int
    original_height: int
    input_width: int
    input_height: int

    # Model configuration
    class_names: List[str]
    thresholds: Dict[str, float]

    # Segmentation outputs
    masks: Dict[str, str]
    probabilities: Dict[str, str]

    # Confidence and uncertainty
    confidence: Dict[str, Optional[float]]
    uncertainty: Dict[str, float]
    image_uncertainty: float

    # Execution information
    device: str
