
import base64
import io
import tempfile
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    Header,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware

from app.inference import (
    ChestXrayInference,
    CLASS_NAMES,
    THRESHOLDS,
    MODEL_VERSION,
)

from app.schemas import (
    ModelInfoResponse,
    PredictionResponse,
)


# --------------------------------------------------
# API KEY AUTHENTICATION
# --------------------------------------------------

# Configure API_KEY as an environment variable.
# Never hardcode the secret or commit it to Git.
API_KEY = os.getenv("API_KEY")


def verify_api_key(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
        description="API key required to access prediction."
    )
):
    """Require a valid API key before running inference."""

    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured."
        )

    if (
        x_api_key is None
        or not secrets.compare_digest(x_api_key, API_KEY)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "API-Key"},
        )

    return True


# --------------------------------------------------
# 1. MODEL INITIALIZATION
# --------------------------------------------------

inference_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine

    print("Initializing Chest X-ray AI model...")

    inference_engine = ChestXrayInference()

    print("API model initialization complete!")

    yield

    inference_engine = None
    print("API shutting down.")


# --------------------------------------------------
# 2. FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Chest X-ray AI Annotation API",
    description=(
        "Human-in-the-loop chest X-ray segmentation "
        "using a five-channel U-Net model with "
        "confidence and uncertainty estimation."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# 3. CORS CONFIGURATION
# --------------------------------------------------

# Set CORS_ORIGINS in the hosting environment to a comma-separated
# list of trusted frontend origins. "*" is suitable only when no
# browser-cookie credentials are used; API-key authentication is header-based.
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# --------------------------------------------------
# 4. IMAGE ENCODING UTILITY
# --------------------------------------------------

def encode_png_base64(array, is_probability=False):
    """
    Convert a 2D NumPy array into a Base64 PNG string.

    Binary masks:
        0 = background
        255 = predicted region

    Probability maps:
        0 = probability 0.0
        255 = probability 1.0
    """

    if is_probability:
        image_array = np.clip(array, 0, 1)
        image_array = (
            image_array * 255
        ).round().astype(np.uint8)

    else:
        image_array = (
            array.astype(np.uint8) * 255
        )

    pil_image = Image.fromarray(image_array)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded


# --------------------------------------------------
# 5. ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Chest X-ray AI Annotation API",
        "status": "running",
        "model_version": MODEL_VERSION,
        "documentation": "/docs",
    }


# --------------------------------------------------
# 6. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
    }


# --------------------------------------------------
# 7. MODEL INFORMATION
# --------------------------------------------------

@app.get(
    "/model-info",
    response_model=ModelInfoResponse
)
def model_info():

    if inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not initialized."
        )

    return ModelInfoResponse(
        model_name="Chest X-ray Multi-label Segmentation",
        architecture="U-Net",
        input_size=[256, 256],
        output_channels=5,
        class_names=CLASS_NAMES,
        thresholds=THRESHOLDS,
        device=str(inference_engine.device),
    )


# --------------------------------------------------
# 8. PREDICTION ENDPOINT
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key),
):

    if inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not initialized."
        )

    # Validate file extension
    filename = file.filename or "uploaded.dicom"
    extension = Path(filename).suffix.lower()

    if extension not in [".dcm", ".dicom"]:
        raise HTTPException(
            status_code=400,
            detail="Only .dcm and .dicom files are supported."
        )

    temporary_path = None

    try:

        # Read uploaded file
        file_content = await file.read()

        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        # Save temporarily for pydicom
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(file_content)
            temporary_path = Path(temp_file.name)

        # Run model inference
        result = inference_engine.predict(temporary_path)

        masks_encoded = {}
        probabilities_encoded = {}

        # Encode each of the five output channels
        for index, class_name in enumerate(CLASS_NAMES):

            masks_encoded[class_name] = encode_png_base64(
                result["masks"][class_name]
            )

            probabilities_encoded[class_name] = encode_png_base64(
                result["probabilities"][index],
                is_probability=True
            )

        # Original DICOM dimensions
        original_height, original_width = result["original_size"]

        # Build prediction response
        return PredictionResponse(
            filename=filename,
            model_version=result["model_version"],

            original_width=original_width,
            original_height=original_height,

            input_width=256,
            input_height=256,

            class_names=result["class_names"],
            thresholds=result["thresholds"],

            masks=masks_encoded,
            probabilities=probabilities_encoded,

            confidence=result["confidence"],
            uncertainty=result["uncertainty"],
            image_uncertainty=result["image_uncertainty"],

            device=result["device"],
        )

    except HTTPException:
        raise

    except Exception as error:

        print("Prediction error:", str(error))

        raise HTTPException(
            status_code=422,
            detail="Prediction failed. Check the server logs for details."
        )

    finally:

        # Remove temporary DICOM file
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

        await file.close()
