
from pathlib import Path
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
    BACKEND_DIR
    / "outputs"
    / "validation_threshold_analysis"
    / "threshold_image_results.csv"
)

OUTPUT_DIR = (
    BACKEND_DIR
    / "outputs"
    / "validation_threshold_analysis"
)

df = pd.read_csv(INPUT_PATH)

AREA_CUTOFFS = [
    1, 10, 25, 50, 100, 150, 250, 500, 750, 1000
]

results = []

for (class_name, threshold), group in df.groupby(
    ["class", "threshold"]
):

    for minimum_area in AREA_CUTOFFS:

        # Ground truth image-level presence
        gt = group["gt_present"].to_numpy(dtype=bool)

        # Apply minimum predicted-area rule
        pred = (
            group["predicted_pixels"].to_numpy()
            >= minimum_area
        )

        tp = int(np.logical_and(gt, pred).sum())
        fp = int(np.logical_and(~gt, pred).sum())
        tn = int(np.logical_and(~gt, ~pred).sum())
        fn = int(np.logical_and(gt, ~pred).sum())

        precision = (
            tp / (tp + fp)
            if tp + fp > 0 else np.nan
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0 else np.nan
        )

        specificity = (
            tn / (tn + fp)
            if tn + fp > 0 else np.nan
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (
                np.isfinite(precision)
                and np.isfinite(recall)
                and precision + recall > 0
            )
            else np.nan
        )

        results.append({
            "class": class_name,
            "probability_threshold": threshold,
            "minimum_area_pixels": minimum_area,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "image_f1": f1,
        })

results_df = pd.DataFrame(results)

output_path = OUTPUT_DIR / "minimum_area_sweep.csv"

results_df.to_csv(output_path, index=False)

# Show results at the previously selected thresholds
selected_thresholds = {
    "Aortic enlargement": 0.60,
    "Cardiomegaly": 0.70,
    "Pleural thickening": 0.10,
    "Pulmonary fibrosis": 0.10,
    "Lung Opacity": 0.15,
}

print("\nMINIMUM-AREA ANALYSIS")
print("=" * 75)

for class_name, threshold in selected_thresholds.items():

    subset = results_df[
        (results_df["class"] == class_name)
        & (
            results_df["probability_threshold"]
            == threshold
        )
    ]

    print(f"\n{class_name} | Probability threshold: {threshold}")

    print(
        subset[[
            "minimum_area_pixels",
            "TP",
            "FP",
            "TN",
            "FN",
            "precision",
            "recall",
            "specificity",
            "image_f1",
        ]].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

print("\nSaved:", output_path)
