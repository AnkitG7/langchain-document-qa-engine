"""Download real-world testing PDFs with proper headers and error handling."""

import os
import sys
import urllib.request
from pathlib import Path

TARGET_DIR = Path("data/real_pdfs")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

PDFS = {
    "transformer_paper.pdf": "https://arxiv.org/pdf/1706.03762",
    "nism_derivatives.pdf": "https://www.intelivisto.com/certification/NISM-Series-VIII%20Equity%20Derivatives%20workbook.pdf",
    "rbi_annual_report.pdf": "https://rbidocs.rbi.org.in/rdocs/AnnualReport/PDFs/0ANNUALREPORT202324_FULLDF549205FA214F62A2441C5320D64A29.PDF",
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def download_file(name: str, url: str):
    target_path = TARGET_DIR / name
    if target_path.exists() and target_path.stat().st_size > 10000:
        print(f"[EXISTS] {name} ({target_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return target_path

    print(f"[DOWNLOADING] {name} from {url}...")
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(target_path, "wb") as out:
            data = resp.read()
            out.write(data)
        print(f"[SUCCESS] {name} downloaded ({target_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return target_path
    except Exception as e:
        print(f"[FAILED] Could not download {name}: {e}")
        return None


if __name__ == "__main__":
    for name, url in PDFS.items():
        download_file(name, url)
