"""Load Interspeech abstracts and convert to LangChain Documents with metadata."""

from typing import List, Dict
from langchain_core.documents import Document

from core import config
from interspeech.scraper import scrape_interspeech_abstracts, filter_dementia_papers


def load_interspeech_abstracts(
    years: List[int] = None,
    filter_dementia: bool = True,
    max_papers: int = None
) -> List[Document]:
    """
    Scrape and load Interspeech abstracts as LangChain Documents.

    Args:
        years: List of years to scrape (default: from config)
        filter_dementia: If True, only keep dementia-related papers
        max_papers: Limit papers per year (for testing)

    Returns:
        List of Document objects with metadata
    """
    if years is None:
        years = config.INTERSPEECH_YEARS

    # Scrape all papers
    papers = scrape_interspeech_abstracts(
        years=years,
        delay_seconds=config.SCRAPE_DELAY,
        max_papers=max_papers
    )

    # Filter for dementia
    if filter_dementia:
        papers = filter_dementia_papers(papers)
        print(f"🔍 Filtered to {len(papers)} dementia-related papers")

    # Convert to LangChain Documents
    documents = []
    for paper in papers:
        # Combine title + abstract as the page content
        content = f"Title: {paper['title']}\nAuthors: {paper['authors']}\nAbstract: {paper['abstract']}"

        doc = Document(
            page_content=content,
            metadata={
                'paper_id': paper['paper_id'],
                'title': paper['title'],
                'authors': paper['authors'],
                'year': paper['year'],
                'url': paper['url'],
                'source_type': 'interspeech_abstract'
            }
        )
        documents.append(doc)

    print(f"✅ Loaded {len(documents)} documents")
    return documents
