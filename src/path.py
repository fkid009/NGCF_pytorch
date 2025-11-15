from pathlib import Path

BASE_DIR = Path(__file__).parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

if __name__ == "__main__":
    print(BASE_DIR)
    print(DATA_DIR)
    print(RAW_DATA_DIR)
    print(PROCESSED_DATA_DIR)