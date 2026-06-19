#!/usr/bin/env python3
"""Download and extract movies.dat from official GroupLens MovieLens 1M dataset."""

import urllib.request
import zipfile
import io
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

def main() -> None:
    dest_dir = Path(__file__).resolve().parents[1] / "configs" / "data"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "movies.dat"

    print(f"Downloading MovieLens 1M zip from {URL}...")
    try:
        req = urllib.request.Request(
            URL, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            zip_data = response.read()
        
        print("Extracting ml-1m/movies.dat...")
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            # List contents to find the exact name
            names = z.namelist()
            movies_file = [n for n in names if n.endswith("movies.dat")]
            if not movies_file:
                raise FileNotFoundError("movies.dat not found in the downloaded zip.")
            
            # Extract and write
            content = z.read(movies_file[0])
            with open(dest_path, "wb") as f:
                f.write(content)
            print(f"Successfully extracted movies.dat -> {dest_path}")
            
    except Exception as e:
        print(f"Error downloading/extracting movies.dat: {e}")

if __name__ == "__main__":
    main()
