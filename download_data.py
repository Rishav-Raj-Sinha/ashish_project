"""Download the Enron-Spam CSV (about 15 MB zipped) into data/ for training.

    python download_data.py
    python train.py

Source: https://github.com/MWiechmann/enron_spam_data, a single-file version of the
Enron-Spam corpus by Metsis, Androutsopoulos and Paliouras (CEAS 2006).
"""

import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip"


def main() -> None:
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    print("Downloading...")
    with urllib.request.urlopen(URL, timeout=120) as response:
        blob = response.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(out)
    print("Saved to", out)


if __name__ == "__main__":
    main()
