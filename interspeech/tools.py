"""Step 5: wrap the retriever as a tool with citations for Interspeech papers."""

from langchain.tools import tool


def create_interspeech_search_tool(retriever):
    """Return a @tool function that searches Interspeech abstracts with citations."""

    @tool
    def search_interspeech(question: str) -> str:
        """
        Search Interspeech paper abstracts for information about speech research.
        Use this tool to look up facts from academic papers presented at Interspeech.
        Returns text chunks with paper citations.
        """
        matching_chunks = retriever.invoke(question)

        results = []
        for chunk in matching_chunks:
            meta = chunk.metadata
            citation = f"【{meta.get('paper_id', 'unknown')}†{meta.get('year', '')}】"
            results.append(f"{chunk.page_content}\n— {citation}")

        return "\n\n".join(results)

    return search_interspeech
