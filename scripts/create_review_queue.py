
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "active_learning"
RESULTS_DIR = DATA_DIR / "results"

RANKING_PATH = RESULTS_DIR / "active_learning_ranked.csv"
QUEUE_PATH = RESULTS_DIR / "human_review_queue.csv"

BATCH_SIZE = 10

# Load acquisition ranking
ranking = pd.read_csv(RANKING_PATH)

# Keep successfully scored images
ranking = ranking[ranking["status"] == "scored"].copy()

# Select highest-uncertainty images
queue = ranking.head(BATCH_SIZE).copy()

# Add human-review workflow fields
queue["review_status"] = "queued_for_review"
queue["reviewer"] = ""
queue["reviewed_at"] = ""
queue["annotation_path"] = ""
queue["review_notes"] = ""

# Save queue
queue.to_csv(QUEUE_PATH, index=False)

print("=" * 50)
print("HUMAN REVIEW QUEUE CREATED")
print("=" * 50)

print(f"Images selected: {len(queue)}")
print(f"Images remaining: {len(ranking) - len(queue)}")
print(f"Saved to: {QUEUE_PATH}")

print("\nSelected images:")
print(
    queue[
        ["rank", "image_id", "acquisition_score", "review_status"]
    ].to_string(index=False)
)
