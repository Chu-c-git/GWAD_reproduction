"""
Download pretrained CIFAR-10 target model weights from huyvnphan/PyTorch_CIFAR10
(https://github.com/huyvnphan/PyTorch_CIFAR10) and keep only the ones GWAD uses.

usage (from repo root):
    python scripts/download_weights.py
"""
import os
import shutil
import tempfile
import zipfile
import truststore
truststore.inject_into_ssl()  # use OS certificate store (works behind antivirus / proxy TLS inspection)
import gdown

# Google Drive backup of state_dicts.zip in huyvnphan/PyTorch_CIFAR10 README (~933MB).
# the Box link used by their train.py returns 404.
GDRIVE_ID = "17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq"
WANTED = ["resnet18.pt"]
OUT_DIR = os.path.join("model", "cifar10")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if all(os.path.exists(os.path.join(OUT_DIR, f)) for f in WANTED):
        print("weights already exist in %s" % OUT_DIR)
        return

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "state_dicts.zip")
        gdown.download(id=GDRIVE_ID, output=zip_path, quiet=False)
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if base in WANTED:
                    with zf.open(name) as src, open(os.path.join(OUT_DIR, base), "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print("extracted %s" % base)


if __name__ == '__main__':
    main()
