from datasets import load_dataset
from pathlib import Path

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load FNSPID from Hugging Face
dataset = load_dataset("Zihan1004/FNSPID")

print(dataset)

# Print available splits
print("Splits:", dataset.keys())

# Take first available split
split_name = list(dataset.keys())[0]
df = dataset[split_name].to_pandas()

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# Save a local CSV sample first
sample_path = OUT_DIR / "fnspid_sample.csv"
df.head(100000).to_csv(sample_path, index=False)

print(f"Saved sample to {sample_path}")
