#!/usr/bin/env python3
"""Download preprocessed MovieLens 1M dataset from hexiangnan/neural_collaborative_filtering."""

import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/hexiangnan/neural_collaborative_filtering/master/Data/"
FILES = [
    "ml-1m.train.rating",
    "ml-1m.test.rating",
    "ml-1m.test.negative",
]

def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {data_dir}")

    for file_name in FILES:
        url = f"{BASE_URL}{file_name}"
        dest_path = data_dir / file_name
        print(f"Downloading {url} -> {dest_path}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"Successfully downloaded {file_name}")
        except Exception as e:
            print(f"Failed to download {file_name}: {e}")

if __name__ == "__main__":
    main()
