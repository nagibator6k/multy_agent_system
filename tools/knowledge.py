from rag.rag import search


def search_knowledge(query: str) -> str:
    """
    Search the educational knowledge base.

    Args:
        query: User's educational question.

    Returns:
        Relevant knowledge context.
    """
    if not query or not query.strip():
        return ""

    return search(query.strip())