# step4_create_dataset.py
# PawGuide AI — LangSmith Custom Dataset Creation
# Step 4: Upload all 11 examples to LangSmith as a reusable dataset

import os
from dotenv import load_dotenv
from langsmith import Client

# Import examples from Step 3
from step3_dataset_examples import examples

load_dotenv()

# ── CONFIGURATION ─────────────────────────────────────────────────────
DATASET_NAME = "pawguide-ai-evaluation-v1"
DATASET_DESCRIPTION = (
    "PawGuide AI evaluation dataset — 11 examples covering dog and cat health queries. "
    "Includes emergency, routine, chronic, home care, toxicology, and image analysis categories. "
    "Designed to evaluate medical accuracy, emergency escalation, medication safety, "
    "and appropriate tone for a German pet health advisory app. "
    "Evaluation language: English. Production language: German and English."
)

# ── CLIENT SETUP ──────────────────────────────────────────────────────
client = Client()

# ── DATASET CREATION ──────────────────────────────────────────────────
def create_dataset():
    # Check if dataset already exists to avoid duplicates
    existing_datasets = list(client.list_datasets())
    existing_names = [d.name for d in existing_datasets]

    if DATASET_NAME in existing_names:
        print(f"Dataset '{DATASET_NAME}' already exists — skipping creation.")
        dataset = next(d for d in existing_datasets if d.name == DATASET_NAME)
    else:
        print(f"Creating dataset: '{DATASET_NAME}'...")
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=DATASET_DESCRIPTION
        )
        print(f"Dataset created. ID: {dataset.id}")

    return dataset


def upload_examples(dataset):
    print(f"\nUploading {len(examples)} examples...")

    # Check how many examples already exist
    existing_examples = list(client.list_examples(dataset_id=dataset.id))

    if len(existing_examples) >= len(examples):
        print(f"Dataset already contains {len(existing_examples)} examples — skipping upload.")
        return

    # Upload all examples
    for ex in examples:
        client.create_example(
            inputs=ex["input"],
            outputs=ex["output"],
            metadata=ex["metadata"],
            dataset_id=dataset.id
        )
        tc_id = ex["metadata"]["test_case_id"]
        print(f"  ✓ Uploaded {tc_id} — {ex['metadata']['category']} / {ex['metadata']['species']}")

    print(f"\nAll {len(examples)} examples uploaded successfully.")


def verify_dataset(dataset):
    print("\nVerifying dataset...")
    uploaded = list(client.list_examples(dataset_id=dataset.id))
    print(f"  Total examples in LangSmith: {len(uploaded)}")
    print()

    for ex in uploaded:
        tc_id = ex.metadata.get("test_case_id", "N/A")
        category = ex.metadata.get("category", "N/A")
        species = ex.metadata.get("species", "N/A")
        query_preview = ex.inputs.get("owner_query", "")[:55] + "..."
        print(f"  {tc_id} | {species:<4} | {category:<14} | {query_preview}")

    print()
    print(f"Dataset URL: https://smith.langchain.com")
    print(f"Look for dataset: '{DATASET_NAME}' in your LangSmith project.")


# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataset = create_dataset()
    upload_examples(dataset)
    verify_dataset(dataset)
    print("\nStep 4 complete. Dataset ready for evaluation.")