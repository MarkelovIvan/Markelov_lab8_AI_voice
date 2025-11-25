# download.py
import os
import tarfile
from urllib.request import urlretrieve

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
OUT = "data"

def download_and_extract(url=URL, out_dir=OUT):
    os.makedirs(out_dir, exist_ok=True)
    archive = os.path.join(out_dir, "LJSpeech-1.1.tar.bz2")
    if not os.path.exists(archive):
        print("Downloading LJ-Speech (~300MB)...")
        urlretrieve(url, archive)
    else:
        print("Archive already present.")
    extracted = os.path.join(out_dir, "LJSpeech-1.1")
    if not os.path.exists(extracted):
        print("Extracting...")
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(out_dir)
        print("Extracted.")
    else:
        print("Already extracted.")
    print("Dataset ready at:", extracted)
    return extracted

if __name__ == "__main__":
    download_and_extract()
