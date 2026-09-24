

# """Scrape paper abstracts from the ISCA Archive (Interspeech conferences)."""

# import time
# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin
# from typing import List, Dict
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import os
# import json

# BASE_URL = "https://www.isca-archive.org"


# def save_papers(papers: List[Dict], filepath: str = "data/interspeech_2020_2025.json"):
#     """Save a list of paper dicts to a JSON file."""
#     with open(filepath, 'w', encoding='utf-8') as f:
#         json.dump(papers, f, indent=2, ensure_ascii=False)
#     print(f"✅ Saved {len(papers)} papers to {filepath}")


# def load_papers(filepath: str = "data/interspeech_2020_2025.json") -> List[Dict]:
#     """Load a list of paper dicts from a JSON file."""
#     try:
#         with open(filepath, 'r', encoding='utf-8') as f:
#             papers = json.load(f)
#         print(f"✅ Loaded {len(papers)} papers from {filepath}")
#         return papers
#     except FileNotFoundError:
#         print(f"❌ File {filepath} not found.")
#         return []

# def get_paper_urls(year: int) -> List[str]:
#     """Get all paper URLs for a given Interspeech year."""
#     index_url = f"{BASE_URL}/interspeech_{year}/index.html"
#     try:
#         # Simple delay before fetching the index page (just to be polite)
#         time.sleep(0.5)
#         resp = requests.get(index_url, timeout=10)
#         resp.raise_for_status()
#     except Exception as e:
#         print(f"❌ Failed to fetch {index_url}: {e}")
#         return []

#     soup = BeautifulSoup(resp.text, 'html.parser')
#     paper_links = []

#     for a in soup.find_all('a', href=True):
#         href = a['href']
#         if href.endswith('.html') and '_interspeech' in href:
#             full_url = urljoin(index_url, href)
#             paper_links.append(full_url)

#     print(f"✅ Found {len(paper_links)} papers for {year}")
#     return paper_links


# def scrape_paper(page_url: str, delay: float = 0.2) -> Dict:
#     """
#     Extract paper metadata and abstract from a paper page.
#     Each thread sleeps for `delay` seconds before making the request.
#     """
#     # Per‑thread delay – this is NOT global, so threads run in parallel
#     time.sleep(delay)

#     try:
#         resp = requests.get(page_url, timeout=10)
#         resp.raise_for_status()
#     except Exception as e:
#         print(f"⚠️ Failed to fetch {page_url}: {e}")
#         return {}

#     soup = BeautifulSoup(resp.text, 'html.parser')

#     # ============================================================
#     # EXTRACT TITLE & AUTHORS
#     # ============================================================
#     title_tag = soup.find('h1')
#     if title_tag:
#         title = title_tag.text.strip()
#     else:
#         page_text = soup.get_text(separator='\n', strip=True)
#         lines = page_text.split('\n')
#         title = lines[0] if lines else "No title"

#     if title.startswith("ISCA Archive - "):
#         title = title.replace("ISCA Archive - ", "", 1)

#     authors = "Unknown"
#     author_tag = soup.find('p', class_='authors')
#     if not author_tag:
#         author_tag = soup.find('div', class_='authors')
#     if author_tag:
#         authors = author_tag.text.strip()
#     else:
#         page_text = soup.get_text(separator='\n', strip=True)
#         lines = page_text.split('\n')
#         if len(lines) > 1:
#             authors = lines[1]

#     # ============================================================
#     # EXTRACT ABSTRACT
#     # ============================================================
#     abstract = ""
#     page_text = soup.get_text(separator='\n', strip=False)

#     if "#### Abstract" in page_text:
#         start = page_text.find("#### Abstract") + len("#### Abstract")
#         end = page_text.find("####", start + 1)
#         if end == -1:
#             end = len(page_text)
#         abstract = page_text[start:end].strip()
#     else:
#         content_div = soup.find('div', id='content')
#         if not content_div:
#             content_div = soup.find('div', class_='content')
#         if content_div:
#             content_text = content_div.get_text(separator='\n', strip=True)
#             lines = [l.strip() for l in content_text.split('\n') if l.strip()]
#             if len(lines) >= 3:
#                 abstract_lines = []
#                 for line in lines[2:]:
#                     if line.startswith(('####', 'Introduction', 'Biography', 'References')):
#                         break
#                     abstract_lines.append(line)
#                 abstract = ' '.join(abstract_lines[:3])

#     if not abstract:
#         if "Abstract" in page_text:
#             start = page_text.find("Abstract") + len("Abstract")
#             abstract = page_text[start:start+1000].strip()
#         else:
#             lines = page_text.split('\n')
#             if len(lines) > 2:
#                 abstract = ' '.join(lines[2:5])

#     # ============================================================
#     # PAPER ID & YEAR
#     # ============================================================
#     paper_id = page_url.split('/')[-1].replace('.html', '')
#     year = page_url.split('/')[2].split('_')[1] if '_' in page_url.split('/')[2] else "unknown"

#     return {
#         'paper_id': paper_id,
#         'title': title,
#         'authors': authors,
#         'abstract': abstract,
#         'year': year,
#         'url': page_url
#     }


# # ============================================================
# # MAIN SCRAPER FUNCTION (Parallel, no global lock)
# # ============================================================
# def scrape_interspeech_abstracts(
#     years: List[int] = [2025],
#     delay_seconds: float = 0.2,
#     max_papers: int = None,
#     max_workers: int = None
# ) -> List[Dict]:
#     """
#     Scrape abstracts from Interspeech conferences in parallel.

#     Args:
#         years: List of years to scrape (e.g., [2024, 2025])
#         delay_seconds: Delay per request per thread (each thread sleeps before its own request)
#         max_papers: Limit number of papers per year (for testing)
#         max_workers: Number of concurrent threads (default: number of CPU cores)

#     Returns:
#         List of paper dicts with 'paper_id', 'title', 'authors', 'abstract', 'year', 'url'
#     """
#     if max_workers is None:
#         max_workers = os.cpu_count() or 8  # Use all available cores

#     all_papers = []

#     for year in years:
#         print(f"\n📚 Scraping Interspeech {year} with {max_workers} workers...")
#         urls = get_paper_urls(year)

#         if max_papers:
#             urls = urls[:max_papers]

#         results = []

#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             # Submit all tasks, passing the per‑thread delay
#             future_to_url = {
#                 executor.submit(scrape_paper, url, delay_seconds): url
#                 for url in urls
#             }

#             for i, future in enumerate(as_completed(future_to_url), 1):
#                 if i % 50 == 0:
#                     print(f"  Progress: {i}/{len(urls)} papers scraped")
#                 paper = future.result()
#                 if paper and paper.get('abstract'):
#                     results.append(paper)

#         # Polite pause between years
#         time.sleep(1)

#         print(f"  ✅ Scraped {len(results)} papers with abstracts for {year}")
#         all_papers.extend(results)

#     print(f"\n✅ Total papers scraped across all years: {len(all_papers)}")
#     return all_papers


# def count_statistics(papers: List[Dict]) -> Dict:
#     """
#     Compute word/character/token statistics for a list of papers.
#     """
#     total_chars = sum(len(p['abstract']) for p in papers)
#     total_words = sum(len(p['abstract'].split()) for p in papers)

#     estimated_tokens = total_chars // 4

#     try:
#         import tiktoken
#         enc = tiktoken.get_encoding("cl100k_base")
#         total_tokens = 0
#         for p in papers:
#             total_tokens += len(enc.encode(p['abstract']))
#         token_method = "tiktoken (accurate)"
#     except ImportError:
#         total_tokens = estimated_tokens
#         token_method = "chars/4 (estimate)"

#     return {
#         'num_papers': len(papers),
#         'total_chars': total_chars,
#         'total_words': total_words,
#         'total_tokens': total_tokens,
#         'token_method': token_method,
#         'estimated_embedding_cost_usd': total_tokens * 0.00002
#     }


# def filter_dementia_papers(papers: List[Dict]) -> List[Dict]:
#     """Filter papers that mention dementia-related keywords."""
#     keywords = ['dementia', 'alzheimer', 'mci', 'cognitive impairment', 'ad', 'mild cognitive']
#     filtered = []
#     for paper in papers:
#         text = (paper['title'] + ' ' + paper['abstract']).lower()
#         if any(kw in text for kw in keywords):
#             filtered.append(paper)
#     return filtered


# # ============================================================
# # MAIN TEST BLOCK
# # ============================================================
# if __name__ == "__main__":
#     from src import config

#     print("=" * 60)
#     print("🧪 Parallel Scraper Test with Statistics")
#     print("=" * 60)

#     years = config.INTERSPEECH_YEARS
#     print(f"\n📅 Scraping years: {years}")

#     # Set max_papers=None to scrape ALL papers, or set a number for testing
#     # max_papers = 20  # <-- Uncomment for a quick test
#     max_papers = None

#     papers = scrape_interspeech_abstracts(
#         years=years,
#         delay_seconds=0.2,
#         max_papers=max_papers,
#         max_workers=16  # <-- Use all logical cores (you have 16)
#     )

#     # --- NEW: Save the scraped data ---
#     save_papers(papers, "data/interspeech_2020_2025.json")

#     print(f"\n📊 Total papers with abstracts: {len(papers)}")

#     stats = count_statistics(papers)
#     print("\n📈 Statistics:")
#     print(f"   Number of papers: {stats['num_papers']:,}")
#     print(f"   Total characters: {stats['total_chars']:,}")
#     print(f"   Total words: {stats['total_words']:,}")
#     print(f"   Total tokens ({stats['token_method']}): {stats['total_tokens']:,}")
#     print(f"   Estimated embedding cost: ${stats['estimated_embedding_cost_usd']:.2f}")

#     dementia_papers = filter_dementia_papers(papers)
#     print(f"\n🧠 Dementia-related papers: {len(dementia_papers)}")
#     if dementia_papers:
#         print("\n   Sample dementia papers:")
#         for i, d in enumerate(dementia_papers[:3], 1):
#             print(f"   {i}. {d['title']}")

#     print("\n✅ Done.")






"""Scrape paper abstracts from the ISCA Archive (Interspeech conferences)."""

import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json, re


BASE_URL = "https://www.isca-archive.org"

# ============================================================
# DATA PERSISTENCE (JSON Caching)
# ============================================================
def get_cache_filepath(years: List[int]) -> str:
    """Generate the cache file path for the given years."""
    years_str = "_".join(str(y) for y in years)
    return f"data/interspeech/interspeech_{years_str}.json"

def save_papers(papers: List[Dict], filepath: str) -> None:
    """Save a list of paper dicts to a JSON file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(papers)} papers to {filepath}")

def load_papers(filepath: str) -> List[Dict]:
    """Load a list of paper dicts from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        print(f"✅ Loaded {len(papers)} papers from {filepath}")
        return papers
    except FileNotFoundError:
        print(f"❌ File {filepath} not found.")
        return []

def papers_exist(filepath: str) -> bool:
    """Check if a cached JSON file exists."""
    return os.path.exists(filepath)

# ============================================================
# SCRAPING FUNCTIONS
# ============================================================
def get_paper_urls(year: int) -> List[str]:
    """Get all paper URLs for a given Interspeech year."""
    index_url = f"{BASE_URL}/interspeech_{year}/index.html"
    try:
        time.sleep(0.5)
        resp = requests.get(index_url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to fetch {index_url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    paper_links = []

    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.endswith('.html') and '_interspeech' in href:
            full_url = urljoin(index_url, href)
            paper_links.append(full_url)

    print(f"✅ Found {len(paper_links)} papers for {year}")
    return paper_links


def scrape_paper(page_url: str, delay: float = 0.2) -> Dict:
    """
    Extract paper metadata and abstract from a paper page.
    Uses meta tags for title and authors, and extracts the abstract from the body.
    """
    time.sleep(delay)

    try:
        resp = requests.get(page_url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"⚠️ Failed to fetch {page_url}: {e}")
        return {}

    soup = BeautifulSoup(resp.text, 'html.parser')

    # ============================================================
    # EXTRACT TITLE (from meta tag)
    # ============================================================
    title_meta = soup.find('meta', {'name': 'citation_title'})
    if title_meta and title_meta.get('content'):
        title = title_meta['content'].strip()
    else:
        # Fallback: try <title> tag
        title_tag = soup.find('title')
        title = title_tag.text.strip() if title_tag else "No title"
        # Remove "ISCA Archive - " prefix if present
        if title.startswith("ISCA Archive - "):
            title = title.replace("ISCA Archive - ", "", 1)

    # ============================================================
    # EXTRACT AUTHORS (from meta tags)
    # ============================================================
    author_metas = soup.find_all('meta', {'name': 'citation_author'})
    if author_metas:
        authors = ", ".join([m.get('content', '').strip() for m in author_metas if m.get('content')])
    else:
        # Fallback: try to find authors in body text
        page_text = soup.get_text(separator='\n', strip=True)
        lines = page_text.split('\n')
        authors = lines[1] if len(lines) > 1 else "Unknown"

    # ============================================================
    # EXTRACT YEAR (FIXED: URL parsing was off by one)
    # ============================================================
    year = "unknown"
    # URL format: https://www.isca-archive.org/interspeech_2020/paper.html
    # Split by '/' and get the part that contains 'interspeech_'
    parts = page_url.split('/')
    for part in parts:
        if part.startswith('interspeech_'):
            year = part.replace('interspeech_', '')
            break

    # ============================================================
    # EXTRACT ABSTRACT (from body text, with noise removal)
    # ============================================================
    abstract = ""
    
    # Get the full text
    body_text = soup.get_text(separator='\n', strip=True)
    lines = [l.strip() for l in body_text.split('\n') if l.strip()]
    
    # Skip the title and author lines
    start_idx = 0
    for i, line in enumerate(lines):
        # Skip lines that look like the title or author list
        if i == 0 and line == title:
            continue
        if i == 1 and line == authors:
            continue
        # Also skip lines that are just the paper ID or metadata
        if line.startswith(('Paper ID:', 'DOI:', 'https://')):
            continue
        # If we find a line that looks like the start of the abstract, use it
        if len(line) > 20 and not line.startswith(('#####', '####')):
            start_idx = i
            break
    
    # Collect the abstract lines (until we hit a BibTeX marker or a short line)
    abstract_lines = []
    for line in lines[start_idx:]:
        # Stop if we hit BibTeX or a reference marker
        if line.startswith(('@', 'author =', 'year =', 'booktitle =')):
            break
        # Stop if we hit a section header like "References" or "Biography"
        if line.lower().startswith(('references', 'biography', 'acknowledgements')):
            break
        # Keep lines that are substantial
        if len(line) > 5:
            abstract_lines.append(line)
    
    # Join and clean the abstract
    if abstract_lines:
        abstract = " ".join(abstract_lines).strip()
        
        # Remove "ISCA Archive - " noise
        abstract = re.sub(r'ISCA Archive\s*-\s*', '', abstract)
        
        # Remove "Archive Interspeech 2020" noise (and similar)
        abstract = re.sub(r'Archive\s+Interspeech\s+\d{4}\s*', '', abstract)
        
        # Remove duplicate spaces
        abstract = re.sub(r'\s+', ' ', abstract).strip()
    
    # If no abstract was found, try the fallback
    if not abstract:
        # Find the position of the first BibTeX-like pattern
        bibtex_start = body_text.find('author =')
        if bibtex_start != -1:
            # Try to find a good start point
            start_pos = 0
            # Try to find the beginning of the abstract after the title/authors
            if authors in body_text:
                start_pos = body_text.find(authors) + len(authors)
            raw_abstract = body_text[start_pos: bibtex_start].strip()
            # Clean it
            raw_abstract = re.sub(r'ISCA Archive\s*-\s*', '', raw_abstract)
            raw_abstract = re.sub(r'Archive\s+Interspeech\s+\d{4}\s*', '', raw_abstract)
            raw_abstract = re.sub(r'\s+', ' ', raw_abstract).strip()
            abstract = raw_abstract
        else:
            # Just take the first few paragraphs after the authors
            lines_after_authors = [l for l in lines if l not in [title, authors] and len(l) > 20]
            if lines_after_authors:
                raw = " ".join(lines_after_authors[:3])
                raw = re.sub(r'ISCA Archive\s*-\s*', '', raw)
                raw = re.sub(r'Archive\s+Interspeech\s+\d{4}\s*', '', raw)
                raw = re.sub(r'\s+', ' ', raw).strip()
                abstract = raw

    # ============================================================
    # PAPER ID
    # ============================================================
    paper_id = page_url.split('/')[-1].replace('.html', '')

    return {
        'paper_id': paper_id,
        'title': title,
        'authors': authors,
        'abstract': abstract,
        'year': year,
        'url': page_url
    }


# ============================================================
# MAIN SCRAPER FUNCTION (Parallel, with caching)
# ============================================================
def scrape_interspeech_abstracts(
    years: List[int] = [2025],
    delay_seconds: float = 0.2,
    max_papers: int = None,
    max_workers: int = None,
    force_scrape: bool = False
) -> List[Dict]:
    """
    Scrape abstracts from Interspeech conferences in parallel.
    If a cached JSON file exists, load from it instead of scraping.

    Args:
        years: List of years to scrape (e.g., [2024, 2025])
        delay_seconds: Delay per request per thread
        max_papers: Limit number of papers per year (for testing)
        max_workers: Number of concurrent threads
        force_scrape: If True, ignore cache and scrape fresh

    Returns:
        List of paper dicts
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 8

    # Check if cached JSON exists
    cache_file = get_cache_filepath(years)
    if not force_scrape and papers_exist(cache_file):
        print(f"📂 Found cached data at {cache_file}")
        return load_papers(cache_file)

    # Otherwise, scrape fresh
    print("🔄 No cache found or force_scrape=True. Scraping fresh data...")
    all_papers = []

    for year in years:
        print(f"\n📚 Scraping Interspeech {year} with {max_workers} workers...")
        urls = get_paper_urls(year)

        if max_papers:
            urls = urls[:max_papers]

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(scrape_paper, url, delay_seconds): url
                for url in urls
            }

            for i, future in enumerate(as_completed(future_to_url), 1):
                if i % 50 == 0:
                    print(f"  Progress: {i}/{len(urls)} papers scraped")
                paper = future.result()
                if paper and paper.get('abstract'):
                    results.append(paper)

        time.sleep(1)  # Polite pause between years

        print(f"  ✅ Scraped {len(results)} papers with abstracts for {year}")
        all_papers.extend(results)

    print(f"\n✅ Total papers scraped across all years: {len(all_papers)}")

    # Save to cache
    save_papers(all_papers, cache_file)

    return all_papers


def count_statistics(papers: List[Dict]) -> Dict:
    """Compute word/character/token statistics for a list of papers."""
    total_chars = sum(len(p['abstract']) for p in papers)
    total_words = sum(len(p['abstract'].split()) for p in papers)

    estimated_tokens = total_chars // 4

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        total_tokens = 0
        for p in papers:
            total_tokens += len(enc.encode(p['abstract']))
        token_method = "tiktoken (accurate)"
    except ImportError:
        total_tokens = estimated_tokens
        token_method = "chars/4 (estimate)"

    return {
        'num_papers': len(papers),
        'total_chars': total_chars,
        'total_words': total_words,
        'total_tokens': total_tokens,
        'token_method': token_method,
        'estimated_embedding_cost_usd': total_tokens * 0.00002
    }


def filter_dementia_papers(papers: List[Dict]) -> List[Dict]:
    """Filter papers that mention dementia-related keywords."""
    keywords = ['dementia', 'alzheimer', 'mci', 'cognitive impairment', 'ad', 'mild cognitive']
    filtered = []
    for paper in papers:
        text = (paper['title'] + ' ' + paper['abstract']).lower()
        if any(kw in text for kw in keywords):
            filtered.append(paper)
    return filtered


# ============================================================
# MAIN TEST BLOCK
# ============================================================
if __name__ == "__main__":
    from core import config

    print("=" * 60)
    print("🧪 Parallel Scraper Test with Caching")
    print("=" * 60)

    years = config.INTERSPEECH_YEARS
    print(f"\n📅 Scraping years: {years}")

    # Set max_papers=None to scrape ALL papers, or set a number for testing
    max_papers = None

    # force_scrape=True will ignore cache and scrape fresh
    # force_scrape=False will load from cache if available
    papers = scrape_interspeech_abstracts(
        years=years,
        delay_seconds=0.2,
        max_papers=max_papers,
        max_workers=16,
        force_scrape=False   # <-- Set to True to force re-scraping
    )

    print(f"\n📊 Total papers with abstracts: {len(papers)}")

    stats = count_statistics(papers)
    print("\n📈 Statistics:")
    print(f"   Number of papers: {stats['num_papers']:,}")
    print(f"   Total characters: {stats['total_chars']:,}")
    print(f"   Total words: {stats['total_words']:,}")
    print(f"   Total tokens ({stats['token_method']}): {stats['total_tokens']:,}")
    print(f"   Estimated embedding cost: ${stats['estimated_embedding_cost_usd']:.2f}")

    dementia_papers = filter_dementia_papers(papers)
    print(f"\n🧠 Dementia-related papers: {len(dementia_papers)}")
    if dementia_papers:
        print("\n   Sample dementia papers:")
        for i, d in enumerate(dementia_papers[:3], 1):
            print(f"   {i}. {d['title']}")

    print("\n✅ Done.")


# if __name__ == "__main__":
#     from src import config

#     print("=" * 60)
#     print("🧪 Parallel Scraper Test with Caching")
#     print("=" * 60)

#     years = config.INTERSPEECH_YEARS
#     print(f"\n📅 Scraping years: {years}")

#     # ============================================================
#     # TEST MODE: Scrape only 5 papers to verify extraction
#     # ============================================================
#     TEST_MODE = True  # Set to False to scrape ALL papers
#     MAX_PAPERS_TEST = 5  # Number of papers to scrape in test mode

#     if TEST_MODE:
#         print(f"\n🧪 TEST MODE: Scraping only {MAX_PAPERS_TEST} papers per year...")
#         max_papers = MAX_PAPERS_TEST
#     else:
#         print(f"\n📚 FULL MODE: Scraping ALL papers...")
#         max_papers = None

#     # force_scrape=True will ignore cache and scrape fresh
#     # force_scrape=False will load from cache if available
#     papers = scrape_interspeech_abstracts(
#         years=years,
#         delay_seconds=0.2,
#         max_papers=max_papers,
#         max_workers=16,
#         force_scrape=False
#     )

#     print(f"\n📊 Total papers scraped: {len(papers)}")

#     # Show sample of first 5 papers
#     if papers:
#         print("\n📄 Sample of first 5 papers:")
#         print("-" * 60)
#         for i, paper in enumerate(papers[:5], 1):
#             print(f"\nPaper {i}:")
#             print(f"   Paper ID: {paper.get('paper_id')}")
#             print(f"   Title: {paper.get('title')}")
#             print(f"   Authors: {paper.get('authors')}")
#             print(f"   Year: {paper.get('year')}")
#             abstract = paper.get('abstract', '')
#             print(f"   Abstract (first 200 chars): {abstract[:200]}...")
#             print(f"   Abstract length: {len(abstract)} chars")
#         print("-" * 60)

#     stats = count_statistics(papers)
#     print("\n📈 Statistics:")
#     print(f"   Number of papers: {stats['num_papers']:,}")
#     print(f"   Total characters: {stats['total_chars']:,}")
#     print(f"   Total words: {stats['total_words']:,}")
#     print(f"   Total tokens ({stats['token_method']}): {stats['total_tokens']:,}")
#     print(f"   Estimated embedding cost: ${stats['estimated_embedding_cost_usd']:.2f}")

#     # Check for empty abstracts
#     empty_abstracts = [p for p in papers if not p.get('abstract') or not p['abstract'].strip()]
#     if empty_abstracts:
#         print(f"\n⚠️ Papers with EMPTY abstracts: {len(empty_abstracts)}")
#         for i, p in enumerate(empty_abstracts[:3], 1):
#             print(f"   {i}. {p.get('title')} (ID: {p.get('paper_id')})")

#     dementia_papers = filter_dementia_papers(papers)
#     print(f"\n🧠 Dementia-related papers: {len(dementia_papers)}")
#     if dementia_papers:
#         print("\n   Sample dementia papers:")
#         for i, d in enumerate(dementia_papers[:3], 1):
#             print(f"   {i}. {d['title']}")

#     print("\n✅ Done.")













# """Test the Bible scraper on a single chapter with noise removal."""

# import requests
# from bs4 import BeautifulSoup
# import re
# import time

# # Test URL: Genesis chapter 1
# URL = "https://wol.jw.org/en/wol/b/r1/lp-e/nwtsty/1/1#study=discover"

# # List of book names to filter out
# BOOK_NAMES = [
#     "genesis", "exodus", "leviticus", "numbers", "deuteronomy",
#     "joshua", "judges", "ruth", "1-samuel", "2-samuel",
#     "1-kings", "2-kings", "1-chronicles", "2-chronicles",
#     "ezra", "nehemiah", "esther", "job", "psalms", "proverbs",
#     "ecclesiastes", "song-of-solomon", "isaiah", "jeremiah",
#     "lamentations", "ezekiel", "daniel", "hosea", "joel",
#     "amos", "obadiah", "jonah", "micah", "nahum", "habakkuk",
#     "zephaniah", "haggai", "zechariah", "malachi", "matthew",
#     "mark", "luke", "john", "acts", "romans", "1-corinthians",
#     "2-corinthians", "galatians", "ephesians", "philippians",
#     "colossians", "1-thessalonians", "2-thessalonians",
#     "1-timothy", "2-timothy", "titus", "philemon", "hebrews",
#     "james", "1-peter", "2-peter", "1-john", "2-john",
#     "3-john", "jude", "revelation"
# ]

# def clean_bible_text(raw_text: str) -> tuple:
#     """
#     Clean the raw Bible text by removing:
#     - Navigation noise
#     - Markers like +, *, †, ‡
#     - Book names
#     - Extra whitespace
#     Returns (cleaned_text, verses_list)
#     """
#     lines = raw_text.split('\n')
    
#     book_names_lower = [b.lower() for b in BOOK_NAMES]
    
#     # First pass: clean each line
#     cleaned_lines = []
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
        
#         # Skip book names
#         if line.lower() in book_names_lower:
#             continue
        
#         # Skip markers that are alone (+, *, †, ‡, etc.)
#         if re.match(r'^[+\*†‡]+$', line):
#             continue
        
#         # Remove markers from the line
#         cleaned_line = re.sub(r'[+\*†‡]', '', line)
#         cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
        
#         if cleaned_line:
#             cleaned_lines.append(cleaned_line)
    
#     # Second pass: parse verses
#     # In JW.org format, verse numbers are often on their own lines
#     verses = []
#     current_verse_num = None
#     current_verse_text = []
    
#     i = 0
#     while i < len(cleaned_lines):
#         line = cleaned_lines[i]
        
#         # Check if this line is a verse number (standalone digit)
#         if line.isdigit() and len(line) <= 3:
#             # This is a verse number
#             if current_verse_num is not None and current_verse_text:
#                 verses.append({
#                     'verse': current_verse_num,
#                     'text': ' '.join(current_verse_text)
#                 })
#             current_verse_num = int(line)
#             current_verse_text = []
#             i += 1
#             # Check if the next line has text (if not, keep collecting)
#             if i < len(cleaned_lines) and cleaned_lines[i].isdigit():
#                 # Next line is also a verse number, so this is just the chapter heading
#                 # or a standalone marker — skip it
#                 current_verse_num = None
#                 continue
#         else:
#             # This is verse text
#             if current_verse_num is not None:
#                 current_verse_text.append(line)
#             i += 1
    
#     # Save the last verse
#     if current_verse_num is not None and current_verse_text:
#         verses.append({
#             'verse': current_verse_num,
#             'text': ' '.join(current_verse_text)
#         })
    
#     # Build clean full text
#     full_text = ""
#     for v in verses:
#         full_text += f"{v['verse']}. {v['text']}\n"
    
#     return full_text.strip(), verses

# def test_scrape():
#     print("=" * 60)
#     print("🧪 Testing Bible scraper on Genesis 1 (with noise removal)")
#     print("=" * 60)

#     print(f"\n1️⃣ Fetching URL: {URL}")
#     try:
#         resp = requests.get(URL, timeout=10)
#         resp.raise_for_status()
#     except Exception as e:
#         print(f"❌ Failed to fetch: {e}")
#         return

#     print(f"   Status code: {resp.status_code}")
#     print(f"   Content length: {len(resp.text)} characters")

#     # 2. Parse with BeautifulSoup
#     soup = BeautifulSoup(resp.text, 'html.parser')

#     # 3. Find the content div
#     print("\n2️⃣ Looking for content divs...")
#     content_div = soup.find('div', id='content')
#     if content_div:
#         print("   ✅ Found div#content")
#     else:
#         content_div = soup.find('div', class_='content')
#         if content_div:
#             print("   ✅ Found div.content")
#         else:
#             print("   ❌ No content div found")
#             return

#     # 4. Extract raw text from content div
#     print("\n3️⃣ Extracting raw text from div#content...")
#     raw_text = content_div.get_text(separator='\n', strip=True)
#     print(f"   Raw text length: {len(raw_text)} characters")
#     print(f"\n   Raw text sample (first 500 chars):\n{raw_text[:500]}...")

#     # 5. Clean the text
#     print("\n4️⃣ Cleaning the text...")
#     clean_text, verses = clean_bible_text(raw_text)
#     print(f"   Clean text length: {len(clean_text)} characters")
#     print(f"   Verses found: {len(verses)}")

#     # 6. Show clean text sample
#     if clean_text:
#         print("\n5️⃣ Clean text sample (first 600 chars):")
#         print("-" * 40)
#         print(clean_text[:600] + ("..." if len(clean_text) > 600 else ""))
#         print("-" * 40)

#         # 7. Show verse breakdown
#         if verses:
#             print("\n6️⃣ Verse breakdown (first 5 verses):")
#             for v in verses[:5]:
#                 print(f"   Verse {v['verse']}: {v['text'][:80]}...")

#         # 8. Statistics
#         words = clean_text.split()
#         chars = len(clean_text)
#         print(f"\n7️⃣ Statistics from clean text:")
#         print(f"   Characters: {chars:,}")
#         print(f"   Words: {len(words):,}")
#         print(f"   Estimated tokens (chars/4): {chars // 4:,}")
#         print(f"   Verses: {len(verses)}")
#     else:
#         print("\n❌ No clean text extracted. Here's what the raw lines look like:")
#         lines = raw_text.split('\n')
#         print(f"   Total lines: {len(lines)}")
#         print(f"   First 30 lines:")
#         for i, line in enumerate(lines[:30]):
#             print(f"   {i+1:2d}. '{line}'")

#     print("\n✅ Done.")

# if __name__ == "__main__":
#     test_scrape()













# """Scrape the Bible from the JW.org website with noise removal and token counting."""

# import time
# import requests
# from bs4 import BeautifulSoup
# import re
# from typing import List, Dict, Tuple
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import os

# BASE_URL = "https://wol.jw.org"

# # Book name → numeric ID mapping
# BOOK_IDS = {
#     "genesis": 1,
#     "exodus": 2,
#     "leviticus": 3,
#     "numbers": 4,
#     "deuteronomy": 5,
#     "joshua": 6,
#     "judges": 7,
#     "ruth": 8,
#     "1-samuel": 9,
#     "2-samuel": 10,
#     "1-kings": 11,
#     "2-kings": 12,
#     "1-chronicles": 13,
#     "2-chronicles": 14,
#     "ezra": 15,
#     "nehemiah": 16,
#     "esther": 17,
#     "job": 18,
#     "psalms": 19,
#     "proverbs": 20,
#     "ecclesiastes": 21,
#     "song-of-solomon": 22,
#     "isaiah": 23,
#     "jeremiah": 24,
#     "lamentations": 25,
#     "ezekiel": 26,
#     "daniel": 27,
#     "hosea": 28,
#     "joel": 29,
#     "amos": 30,
#     "obadiah": 31,
#     "jonah": 32,
#     "micah": 33,
#     "nahum": 34,
#     "habakkuk": 35,
#     "zephaniah": 36,
#     "haggai": 37,
#     "zechariah": 38,
#     "malachi": 39,
#     "matthew": 40,
#     "mark": 41,
#     "luke": 42,
#     "john": 43,
#     "acts": 44,
#     "romans": 45,
#     "1-corinthians": 46,
#     "2-corinthians": 47,
#     "galatians": 48,
#     "ephesians": 49,
#     "philippians": 50,
#     "colossians": 51,
#     "1-thessalonians": 52,
#     "2-thessalonians": 53,
#     "1-timothy": 54,
#     "2-timothy": 55,
#     "titus": 56,
#     "philemon": 57,
#     "hebrews": 58,
#     "james": 59,
#     "1-peter": 60,
#     "2-peter": 61,
#     "1-john": 62,
#     "2-john": 63,
#     "3-john": 64,
#     "jude": 65,
#     "revelation": 66,
# }

# # Chapter counts per book (New World Translation)
# CHAPTER_COUNTS = {
#     "genesis": 50,
#     "exodus": 40,
#     "leviticus": 27,
#     "numbers": 36,
#     "deuteronomy": 34,
#     "joshua": 24,
#     "judges": 21,
#     "ruth": 4,
#     "1-samuel": 31,
#     "2-samuel": 24,
#     "1-kings": 22,
#     "2-kings": 25,
#     "1-chronicles": 29,
#     "2-chronicles": 36,
#     "ezra": 10,
#     "nehemiah": 13,
#     "esther": 10,
#     "job": 42,
#     "psalms": 150,
#     "proverbs": 31,
#     "ecclesiastes": 12,
#     "song-of-solomon": 8,
#     "isaiah": 66,
#     "jeremiah": 52,
#     "lamentations": 5,
#     "ezekiel": 48,
#     "daniel": 12,
#     "hosea": 14,
#     "joel": 3,
#     "amos": 9,
#     "obadiah": 1,
#     "jonah": 4,
#     "micah": 7,
#     "nahum": 3,
#     "habakkuk": 3,
#     "zephaniah": 3,
#     "haggai": 2,
#     "zechariah": 14,
#     "malachi": 4,
#     "matthew": 28,
#     "mark": 16,
#     "luke": 24,
#     "john": 21,
#     "acts": 28,
#     "romans": 16,
#     "1-corinthians": 16,
#     "2-corinthians": 13,
#     "galatians": 6,
#     "ephesians": 6,
#     "philippians": 4,
#     "colossians": 4,
#     "1-thessalonians": 5,
#     "2-thessalonians": 3,
#     "1-timothy": 6,
#     "2-timothy": 4,
#     "titus": 3,
#     "philemon": 1,
#     "hebrews": 13,
#     "james": 5,
#     "1-peter": 5,
#     "2-peter": 3,
#     "1-john": 5,
#     "2-john": 1,
#     "3-john": 1,
#     "jude": 1,
#     "revelation": 22,
# }

# BOOK_NAMES = list(BOOK_IDS.keys())


# def get_chapter_url(book_name: str, chapter: int) -> str:
#     return f"{BASE_URL}/en/wol/b/r1/lp-e/nwtsty/{BOOK_IDS[book_name]}/{chapter}#study=discover"


# def clean_bible_text(raw_text: str) -> Tuple[str, List[Dict]]:
#     """Clean raw Bible text and extract verses."""
#     lines = raw_text.split('\n')
#     book_names_lower = [b.lower() for b in BOOK_NAMES]

#     # First pass: clean each line
#     cleaned_lines = []
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#         if line.lower() in book_names_lower:
#             continue
#         if re.match(r'^[+\*†‡]+$', line):
#             continue
#         cleaned_line = re.sub(r'[+\*†‡]', '', line)
#         cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
#         if cleaned_line:
#             cleaned_lines.append(cleaned_line)

#     # Second pass: parse verses
#     verses = []
#     current_verse_num = None
#     current_verse_text = []

#     i = 0
#     while i < len(cleaned_lines):
#         line = cleaned_lines[i]
#         if line.isdigit() and len(line) <= 3:
#             if current_verse_num is not None and current_verse_text:
#                 verses.append({
#                     'verse': current_verse_num,
#                     'text': ' '.join(current_verse_text)
#                 })
#             current_verse_num = int(line)
#             current_verse_text = []
#             i += 1
#             # Skip if next line is also a digit (chapter heading)
#             if i < len(cleaned_lines) and cleaned_lines[i].isdigit():
#                 current_verse_num = None
#                 continue
#         else:
#             if current_verse_num is not None:
#                 current_verse_text.append(line)
#             i += 1

#     if current_verse_num is not None and current_verse_text:
#         verses.append({
#             'verse': current_verse_num,
#             'text': ' '.join(current_verse_text)
#         })

#     full_text = "\n".join(f"{v['verse']}. {v['text']}" for v in verses)
#     return full_text.strip(), verses


# def scrape_chapter(page_url: str, delay: float = 0.5) -> Dict:
#     """Scrape a single chapter and return cleaned data."""
#     time.sleep(delay)
#     try:
#         resp = requests.get(page_url, timeout=15)
#         resp.raise_for_status()
#     except Exception as e:
#         print(f"⚠️ Failed to fetch {page_url}: {e}")
#         return {}

#     soup = BeautifulSoup(resp.text, 'html.parser')
#     content_div = soup.find('div', id='content')
#     if not content_div:
#         content_div = soup.find('div', class_='content')
#     if not content_div:
#         return {}

#     raw_text = content_div.get_text(separator='\n', strip=True)
#     clean_text, verses = clean_bible_text(raw_text)

#     # Extract book and chapter from URL
#     parts = page_url.split('/')
#     book_num = parts[6] if len(parts) > 6 else "unknown"
#     chapter_num = parts[7] if len(parts) > 7 else "unknown"

#     # Find book name from mapping
#     book_name = next((b for b, n in BOOK_IDS.items() if str(n) == book_num), "unknown")

#     return {
#         'book': book_name,
#         'book_num': int(book_num) if book_num.isdigit() else 0,
#         'chapter': int(chapter_num) if chapter_num.isdigit() else 0,
#         'verses': verses,
#         'full_text': clean_text,
#         'url': page_url
#     }


# def scrape_bible(
#     books: List[str] = None,
#     delay_seconds: float = 0.5,
#     max_workers: int = 8,
#     max_chapters: int = None
# ) -> List[Dict]:
#     """Scrape the Bible (parallel) and return list of chapter dicts."""
#     if books is None:
#         books = list(BOOK_IDS.keys())

#     all_urls = []
#     for book in books:
#         if book not in CHAPTER_COUNTS:
#             continue
#         for ch in range(1, CHAPTER_COUNTS[book] + 1):
#             all_urls.append((book, ch, get_chapter_url(book, ch)))

#     if max_chapters:
#         all_urls = all_urls[:max_chapters]

#     print(f"📚 Total chapters to scrape: {len(all_urls)}")

#     results = []
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_info = {
#             executor.submit(scrape_chapter, url, delay_seconds): (book, ch, url)
#             for book, ch, url in all_urls
#         }
#         for i, future in enumerate(as_completed(future_to_info), 1):
#             if i % 20 == 0:
#                 print(f"  Progress: {i}/{len(all_urls)} chapters")
#             data = future.result()
#             if data and data.get('full_text'):
#                 results.append(data)

#     print(f"✅ Scraped {len(results)} chapters successfully")
#     return results


# def count_statistics(chapters: List[Dict]) -> Dict:
#     """Compute character/word/token statistics for scraped chapters."""
#     total_chars = sum(len(c['full_text']) for c in chapters)
#     total_words = sum(len(c['full_text'].split()) for c in chapters)
#     total_verses = sum(len(c['verses']) for c in chapters)

#     estimated_tokens = total_chars // 4

#     try:
#         import tiktoken
#         enc = tiktoken.get_encoding("cl100k_base")
#         total_tokens = 0
#         for c in chapters:
#             total_tokens += len(enc.encode(c['full_text']))
#         token_method = "tiktoken (accurate)"
#     except ImportError:
#         total_tokens = estimated_tokens
#         token_method = "chars/4 (estimate)"

#     return {
#         'num_chapters': len(chapters),
#         'total_verses': total_verses,
#         'total_chars': total_chars,
#         'total_words': total_words,
#         'total_tokens': total_tokens,
#         'token_method': token_method,
#         'estimated_embedding_cost_usd': total_tokens * 0.00002
#     }


# if __name__ == "__main__":
#     print("=" * 60)
#     print("🧪 Full Bible Scraper Test (Limited)")
#     print("=" * 60)

#     # Test with a few books to estimate token count
#     test_books = ["genesis", "matthew", "psalms"]  # Diverse: OT, NT, poetry
#     max_chapters = 20  # Limit to 20 chapters for quick test

#     print(f"\n📚 Books: {test_books}")
#     print(f"📄 Chapters limit: {max_chapters} (per book, but we apply a global cap)")

#     chapters = scrape_bible(
#         books=test_books,
#         delay_seconds=0.5,
#         max_workers=8,
#         max_chapters=max_chapters
#     )

#     stats = count_statistics(chapters)
#     print("\n📈 Statistics:")
#     print(f"   Chapters: {stats['num_chapters']}")
#     print(f"   Verses: {stats['total_verses']:,}")
#     print(f"   Characters: {stats['total_chars']:,}")
#     print(f"   Words: {stats['total_words']:,}")
#     print(f"   Tokens ({stats['token_method']}): {stats['total_tokens']:,}")
#     print(f"   Estimated embedding cost: ${stats['estimated_embedding_cost_usd']:.2f}")

#     # Estimate full Bible
#     total_bible_chapters = sum(CHAPTER_COUNTS.values())
#     avg_tokens_per_chapter = stats['total_tokens'] / stats['num_chapters'] if stats['num_chapters'] else 0
#     estimated_full_tokens = avg_tokens_per_chapter * total_bible_chapters
#     print(f"\n📊 Estimated full Bible tokens: {estimated_full_tokens:,.0f} (~{estimated_full_tokens/1e6:.1f}M)")
#     if estimated_full_tokens > 1_000_000:
#         print("⚠️ This exceeds Jina's 1M token free tier. Consider using a subset or another provider.")

#     print("\n✅ Done.")