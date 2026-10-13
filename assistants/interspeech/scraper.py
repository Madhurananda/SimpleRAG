
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
    from assistants.interspeech import config

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
