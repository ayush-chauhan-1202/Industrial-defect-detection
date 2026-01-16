import os
import urllib.request
import tarfile
from tqdm import tqdm

MVTEC_URL = "https://www.mvtec.com/fileadmin/Redaktion/Industrie_und_Forschung/Images/Datasets/MVTec_AD/mvtec_anomaly_detection.tar.xz"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARCHIVE_PATH = os.path.join(DATA_DIR, "mvtec.tar.xz")
EXTRACT_DIR = os.path.join(DATA_DIR, "mvtec")

class ProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_mvtec():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(EXTRACT_DIR):
        print(f"✓ MVTec dataset already exists at {EXTRACT_DIR}")
        return

    print("Downloading MVTec AD dataset...")

    with ProgressBar(unit='B', unit_scale=True, desc="MVTec") as bar:
        urllib.request.urlretrieve(MVTEC_URL, ARCHIVE_PATH, reporthook=bar.update_to)

    print("Extracting archive...")
    with tarfile.open(ARCHIVE_PATH, "r:xz") as tar:
        tar.extractall(DATA_DIR)

    os.rename(
        os.path.join(DATA_DIR, "mvtec_anomaly_detection"),
        EXTRACT_DIR
    )

    print("✓ Dataset ready")
