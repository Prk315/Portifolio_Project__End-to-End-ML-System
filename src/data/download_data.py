import os
import pandas as pd
import requests
from pathlib import Path

def download_lending_club_data():
    """
    Downloads Lending Club loan data.
    Note: You'll need to download from Kaggle manually or use Kaggle API.
    """
    print("=" * 60)
    print("LENDING CLUB DATA DOWNLOAD")
    print("=" * 60)

    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\nTo download the dataset:")
    print("1. Visit: https://www.kaggle.com/datasets/wordsforthewise/lending-club")
    print("2. Download 'accepted_2007_to_2018Q4.csv.gz'")
    print(f"3. Place it in: {data_dir}/")
    print("\nOr use Kaggle API:")
    print("   kaggle datasets download -d wordsforthewise/lending-club")
    print("   unzip lending-club.zip -d data/raw/")

    return data_dir

def download_credit_default_data():
    """
    Downloads UCI Credit Card Default dataset.
    """
    print("=" * 60)
    print("CREDIT DEFAULT DATA DOWNLOAD")
    print("=" * 60)

    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    output_path = data_dir / "credit_default.xls"

    try:
        print(f"\nDownloading from {url}...")
        response = requests.get(url)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"Downloaded to: {output_path}")

        df = pd.read_excel(output_path, skiprows=1)
        csv_path = data_dir / "credit_default.csv"
        df.to_csv(csv_path, index=False)
        print(f"Converted to CSV: {csv_path}")

        return csv_path

    except Exception as e:
        print(f"Error downloading data: {e}")
        print("\nAlternative: Download manually from:")
        print("https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients")
        return None

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "credit":
        download_credit_default_data()
    else:
        download_lending_club_data()
