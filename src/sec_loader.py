from __future__ import annotations
import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import COMPANIES, YEARS, FILING_TYPE, SEC_HEADERS, RAW_DIR, PROCESSED_DIR


def _search_filings_page(page: dict, ticker: str, cik: str, company_name: str, year: int, filing_type: str) -> list:
    """
    Scan one page of the EDGAR filings array and return 10-Ks for fiscal year ``year``.

    The fiscal year is identified by the filing's ``reportDate`` (the fiscal
    period end), NOT its filing date.  Filing-date heuristics break for
    companies whose fiscal year does not end in December: Apple (late
    September, filed ~November of the same year) and Microsoft (June, filed
    ~July) were previously mapped one year off.  NVIDIA's fiscal year ends in
    late January, so its "fiscal 2024" 10-K has reportDate 2024-01-28.
    """
    forms = page.get("form", [])
    dates = page.get("filingDate", [])
    report_dates = page.get("reportDate", [""] * len(forms))
    accessions = page.get("accessionNumber", [])
    primary_docs = page.get("primaryDocument", [""] * len(forms))
    candidates = []
    for form, date, rdate, accession, pdoc in zip(forms, dates, report_dates, accessions, primary_docs):
        if form != filing_type or not rdate:
            continue            # skips amendments (10-K/A) and filings without a period
        if int(rdate[:4]) == year:
            candidates.append({
                "ticker": ticker,
                "cik": cik,
                "company": company_name,
                "year": year,
                "filing_type": filing_type,
                "accession_number": accession,
                "filing_date": date,
                "report_date": rdate,
                "primary_document": pdoc,
            })
    return candidates


def get_filing_metadata(ticker: str, cik: str, year: int, filing_type: str = "10-K") -> dict:
    """
    Fetch the accession number and filing date of the 10-K whose fiscal period
    ends in ``year`` (matched on EDGAR ``reportDate``).
    Paginates through additional filing pages for prolific filers (e.g. JPM).
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=SEC_HEADERS, timeout=30)
    response.raise_for_status()
    data = response.json()

    company_name = data.get("name", COMPANIES.get(ticker, {}).get("name", ticker))

    # Search the primary recent page
    candidates = _search_filings_page(
        data.get("filings", {}).get("recent", {}),
        ticker, cik, company_name, year, filing_type
    )

    # If not found, paginate through additional filing index files
    if not candidates:
        extra_files = data.get("filings", {}).get("files", [])
        for extra in extra_files:
            name = extra.get("name", "")
            if not name:
                continue
            page_url = f"https://data.sec.gov/submissions/{name}"
            time.sleep(0.1)
            page_resp = requests.get(page_url, headers=SEC_HEADERS, timeout=30)
            if page_resp.status_code != 200:
                continue
            page_data = page_resp.json()
            candidates = _search_filings_page(page_data, ticker, cik, company_name, year, filing_type)
            if candidates:
                break

    # If more than one original 10-K reports the same fiscal year (rare), keep the latest filing
    chosen = max(candidates, key=lambda c: c["filing_date"]) if candidates else None

    if chosen is None:
        raise ValueError(f"No {filing_type} found for {ticker} (CIK {cik}) fiscal year {year}")

    return chosen


def download_filing_text(metadata: dict, save_dir: str) -> str:
    """
    Download the primary filing document for a given accession number,
    strip HTML tags, and save as plain text.
    """
    os.makedirs(save_dir, exist_ok=True)

    accession_clean = metadata["accession_number"].replace("-", "")
    accession_dashed = metadata["accession_number"]
    cik = metadata["cik"].lstrip("0")

    index_url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={metadata['cik']}&type={metadata['filing_type']}"
        f"&dateb=&owner=include&count=40"
    )

    # Use the filing index JSON to find the primary document
    filing_index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik}"
        f"/{accession_clean}/{accession_dashed}-index.json"
    )

    resp = requests.get(filing_index_url, headers=SEC_HEADERS, timeout=30)
    if resp.status_code != 200:
        # Fall back to HTML index
        filing_index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}"
            f"/{accession_clean}/{accession_dashed}-index.htm"
        )
        resp = requests.get(filing_index_url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        primary_doc = _find_primary_doc_from_html_index(soup, metadata["filing_type"])
    else:
        index_data = resp.json()
        primary_doc = _find_primary_doc_from_json_index(index_data)

    if primary_doc is None:
        raise ValueError(
            f"Could not locate primary document for {metadata['ticker']} {metadata['year']}"
        )

    doc_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik}"
        f"/{accession_clean}/{primary_doc}"
    )

    time.sleep(0.1)
    doc_resp = requests.get(doc_url, headers=SEC_HEADERS, timeout=60)
    doc_resp.raise_for_status()

    content_type = doc_resp.headers.get("Content-Type", "")
    raw_content = doc_resp.text

    if "html" in content_type.lower() or primary_doc.endswith((".htm", ".html")):
        soup = BeautifulSoup(raw_content, "html.parser")
        # Remove script and style elements
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    else:
        text = raw_content

    # Normalize whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    filename = f"{metadata['ticker']}_{metadata['year']}_10K.txt"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    return filepath


def _find_primary_doc_from_json_index(index_data: dict) -> str | None:
    """Extract the primary document filename from a filing index JSON."""
    directory = index_data.get("directory", {})
    items = directory.get("item", [])

    # Prefer the document with type matching 10-K
    for item in items:
        if isinstance(item, dict):
            item_type = item.get("type", "")
            name = item.get("name", "")
            if item_type in ("10-K", "10-K405") and name.endswith((".htm", ".txt", ".html")):
                return name

    # Fall back to first .htm file that isn't an exhibit
    for item in items:
        if isinstance(item, dict):
            name = item.get("name", "")
            if name.endswith((".htm", ".html")) and "ex" not in name.lower():
                return name

    # Last resort: any .htm
    for item in items:
        if isinstance(item, dict):
            name = item.get("name", "")
            if name.endswith((".htm", ".html", ".txt")):
                return name

    return None


def _find_primary_doc_from_html_index(soup: BeautifulSoup, filing_type: str) -> str | None:
    """Extract the primary document filename from an HTML filing index page."""
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 4:
            doc_type = cells[3].get_text(strip=True)
            if doc_type in (filing_type, "10-K405"):
                link = cells[2].find("a")
                if link and link.get("href"):
                    href = link["href"]
                    return href.split("/")[-1]

    # Fallback: first link ending in .htm
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith((".htm", ".html")) and "ex" not in href.lower():
            return href.split("/")[-1]

    return None


def load_all_filings(save_dir: str = None) -> pd.DataFrame:
    """
    Download all 10-K filings for all configured companies and years.
    Saves metadata to data/processed/filing_metadata.csv and returns it.
    """
    if save_dir is None:
        save_dir = RAW_DIR

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    records = []
    tasks = [(ticker, info, year) for ticker, info in COMPANIES.items() for year in YEARS]

    for ticker, info, year in tqdm(tasks, desc="Downloading filings"):
        try:
            print(f"  Fetching metadata: {ticker} {year}")
            metadata = get_filing_metadata(ticker, info["cik"], year, FILING_TYPE)
            time.sleep(0.15)

            print(f"  Downloading: {ticker} {year} ({metadata['accession_number']})")
            local_path = download_filing_text(metadata, save_dir)
            time.sleep(0.15)

            metadata["local_path"] = local_path
            records.append(metadata)
            print(f"  Saved: {local_path}")

        except Exception as e:
            print(f"  ERROR {ticker} {year}: {e}")
            records.append({
                "ticker": ticker,
                "cik": info["cik"],
                "company": info["name"],
                "year": year,
                "filing_type": FILING_TYPE,
                "accession_number": None,
                "filing_date": None,
                "local_path": None,
                "error": str(e),
            })

    df = pd.DataFrame(records)
    metadata_path = os.path.join(PROCESSED_DIR, "filing_metadata.csv")
    df.to_csv(metadata_path, index=False)
    print(f"\nMetadata saved to {metadata_path}")

    return df
