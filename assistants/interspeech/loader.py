"""Load Interspeech abstracts and convert to LangChain Documents with metadata."""

from typing import List

from langchain_core.documents import Document

from assistants.interspeech import config
from assistants.interspeech.scraper import scrape_interspeech_abstracts, filter_dementia_papers


def load_interspeech_abstracts(
    years: List[int] = None,
    filter_dementia: bool = True,
    max_papers: int = None,
) -> List[Document]:
    """Scrape and load Interspeech abstracts as LangChain Documents."""
    if years is None:
        years = config.INTERSPEECH_YEARS

    papers = scrape_interspeech_abstracts(
        years=years,
        delay_seconds=config.SCRAPE_DELAY,
        max_papers=max_papers,
    )

    if filter_dementia:
        papers = filter_dementia_papers(papers)
        print(f"🔍 Filtered to {len(papers)} dementia-related papers")

    documents = []
    for paper in papers:
        content = (
            f"Title: {paper['title']}\n"
            f"Authors: {paper['authors']}\n"
            f"Abstract: {paper['abstract']}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "paper_id":   paper["paper_id"],
                "title":      paper["title"],
                "authors":    paper["authors"],
                "year":       paper["year"],
                "url":        paper["url"],
                "source_type": "interspeech_abstract",
            },
        )
        documents.append(doc)

    print(f"✅ Loaded {len(documents)} documents")
    return documents