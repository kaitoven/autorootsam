import argparse
import os
import urllib.request

DEFAULT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"


def _progress(block_num: int, block_size: int, total_size: int):
    downloaded = block_num * block_size
    if total_size <= 0:
        print(f"Downloaded {downloaded / (1024 * 1024):.1f} MB", end="\r")
        return
    pct = min(100.0, downloaded * 100.0 / total_size)
    mb = downloaded / (1024 * 1024)
    tmb = total_size / (1024 * 1024)
    print(f"{pct:6.2f}%  {mb:8.1f}/{tmb:8.1f} MB", end="\r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", default="checkpoints/sam2.1_hiera_large.pt")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"Downloading:\n  {args.url}\n-> {args.out}")
    urllib.request.urlretrieve(args.url, args.out, reporthook=_progress)
    print("\nDone.")


if __name__ == "__main__":
    main()
