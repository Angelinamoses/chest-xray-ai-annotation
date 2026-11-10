
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_DIR))

# Pylance cannot resolve this package at analysis time because the backend root is
# added dynamically at runtime; suppress the missing-import warning for this script.
from app.inference import ChestXrayInference, CLASS_NAMES  # type: ignore[reportMissingImports]
from app.active_learning import calculate_acquisition_score  # type: ignore[reportMissingImports]


DATA_DIR = PROJECT_ROOT / "data" / "active_learning"
DICOM_DIR = DATA_DIR / "dicom"
MANIFEST_PATH = DATA_DIR / "active_learning_pool_manifest.csv"
OUTPUT_DIR = DATA_DIR / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "active_learning_ranked.csv"


def main():
    print("=" * 55)
    print("ACTIVE LEARNING: BATCH ACQUISITION SCORING")
    print("=" * 55)

    # Load manifest
    manifest = pd.read_csv(MANIFEST_PATH)

    print(f"\nImages in manifest: {len(manifest)}")

    # Load trained model once
    print("\nLoading trained U-Net...")
    model = ChestXrayInference()
    print("Model loaded successfully!")

    results = []

    for index, row in manifest.iterrows():

        image_id = str(row["image_id"])
        dicom_path = DICOM_DIR / f"{image_id}.dicom"

        print(f"\n[{index + 1}/{len(manifest)}] Processing {image_id}")

        if not dicom_path.exists():
            print("ERROR: DICOM file not found.")

            results.append({
                "image_id": image_id,
                "status": "missing_file"
            })
            continue

        try:
            # Run model inference
            prediction = model.predict(str(dicom_path))

            probabilities = prediction["probabilities"]

            # Calculate acquisition score
            score_result = calculate_acquisition_score(
                probabilities
            )

            class_uncertainty = score_result["class_uncertainty"]

            result = {
                "image_id": image_id,
                "filename": dicom_path.name,
                "acquisition_score": score_result["acquisition_score"],
                "scoring_method": score_result["scoring_method"],
                "model_version": prediction["model_version"],
                "status": "scored",
                "scored_at_utc": datetime.now(timezone.utc).isoformat()
            }

            # Add uncertainty for each finding
            for class_name in CLASS_NAMES:
                result[f"{class_name}_uncertainty"] = (
                    class_uncertainty[class_name]
                )

            results.append(result)

            print(
                f"Acquisition score: "
                f"{score_result['acquisition_score']:.5f}"
            )

        except Exception as error:
            print(f"ERROR: {error}")

            results.append({
                "image_id": image_id,
                "filename": dicom_path.name,
                "status": "inference_failed",
                "error": str(error)
            })

    # Create ranking
    results_df = pd.DataFrame(results)

    if "acquisition_score" not in results_df.columns:
        raise RuntimeError("No images were successfully scored.")

    results_df = results_df.sort_values(
        by="acquisition_score",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    results_df.insert(
        0,
        "rank",
        range(1, len(results_df) + 1)
    )

    results_df.to_csv(OUTPUT_PATH, index=False)

    successful = (results_df["status"] == "scored").sum()
    failed = len(results_df) - successful

    print("\n" + "=" * 55)
    print("BATCH SCORING SUMMARY")
    print("=" * 55)

    print(f"Total images: {len(results_df)}")
    print(f"Successfully scored: {successful}")
    print(f"Failed: {failed}")
    print(f"\nSaved ranking to:\n{OUTPUT_PATH}")

    print("\nTOP 10 MOST UNCERTAIN IMAGES:")
    print(
        results_df[
            ["rank", "image_id", "acquisition_score", "status"]
        ].head(10).to_string(index=False)
    )

    print("\nScoring completed.")


if __name__ == "__main__":
    main()
