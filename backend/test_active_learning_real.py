
from pathlib import Path

from app.inference import ChestXrayInference
from app.active_learning import calculate_acquisition_score


# --------------------------------------------------
# 1. LOAD A REAL DICOM
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DICOM_PATH = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "train"
    / "vindr_300"
    / "020717d6ab0b440b37978d9bace9f9b2.dicom"
)

if not DICOM_PATH.exists():
    raise FileNotFoundError(
        f"DICOM not found: {DICOM_PATH}"
    )


# --------------------------------------------------
# 2. LOAD MODEL AND RUN INFERENCE
# --------------------------------------------------

print("\nLoading U-Net model...")

engine = ChestXrayInference()

print("Running inference...")

result = engine.predict(DICOM_PATH)


# --------------------------------------------------
# 3. CALCULATE ACQUISITION SCORE
# --------------------------------------------------

probabilities = result["probabilities"]

scoring_result = calculate_acquisition_score(
    probabilities
)


# --------------------------------------------------
# 4. DISPLAY RESULTS
# --------------------------------------------------

print("\nACTIVE LEARNING RESULTS")
print("--------------------------------")

print(
    f"Image acquisition score: "
    f"{scoring_result['acquisition_score']:.6f}"
)

print("\nClass-wise uncertainty:")

for class_name, score in (
    scoring_result["class_uncertainty"].items()
):

    print(f"{class_name}: {score:.6f}")

print(
    "\nScoring method:",
    scoring_result["scoring_method"]
)

print("--------------------------------")
print("REAL IMAGE SCORING COMPLETED!")
