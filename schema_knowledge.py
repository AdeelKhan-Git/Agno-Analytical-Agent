import json
import logging
import os

from agno.knowledge.knowledge import Knowledge
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType

logger = logging.getLogger(__name__)

LANCEDB_URI = "./data/lancedb"
EMBEDDER_MODEL = "nomic-embed-text"


TABLE_NAME = "schema_knowledge"

GLOSSARY_PATH = os.path.join(os.path.dirname(__file__), "column_glossary.json")
DESCRIPTIONS_PATH = os.path.join(os.path.dirname(__file__), "column_descriptions.json")
TABLE_DESCRIPTIONS_PATH = os.path.join(os.path.dirname(__file__), "table_descriptions.json")


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def build_knowledge() -> Knowledge:
    """Constructs the Knowledge object. Called both by rebuild_index.py
    (to populate it) and by glossary_tools.py (to search it) — same
    construction, so search always points at the same LanceDb table the
    index was actually built into."""
    vector_db = LanceDb(
        uri=LANCEDB_URI,
        table_name=TABLE_NAME,
        search_type=SearchType.hybrid,  # vector + keyword — glossary/column
                                         # text is short, so exact term
                                         # matches (e.g. an exact column
                                         # name) benefit from the keyword
                                         # half as much as semantic phrasing
                                         # benefits from the vector half
        embedder=OllamaEmbedder(
            id=EMBEDDER_MODEL,
            dimensions=768,
        ),
    )
    return Knowledge(vector_db=vector_db, max_results=5)


def _iter_documents():
    """Yields (name, text_content) pairs from all three JSON files —
    kept as small, focused chunks (one per table description, one per
    column description, one per glossary entry) rather than one giant
    blob per table, so semantic search can pinpoint the specific
    relevant fact."""
    table_descriptions = _load_json(TABLE_DESCRIPTIONS_PATH)
    for table, description in table_descriptions.items():
        yield f"{table} (table overview)", f"Table {table}: {description}"

    column_descriptions = _load_json(DESCRIPTIONS_PATH)
    for table, columns in column_descriptions.items():
        for column, description in columns.items():
            yield f"{table}.{column} (column description)", f"Table {table}, column {column}: {description}"

    glossary = _load_json(GLOSSARY_PATH)
    for qualified_col, mapping in glossary.items():
        parts = qualified_col.rsplit(".", 1)
        if len(parts) != 2:
            continue
        table, column = parts
        meaning = mapping.get("meaning", "")
        codes = {k: v for k, v in mapping.items() if k != "meaning"}
        codes_text = "; ".join(f"{k} = {v}" for k, v in codes.items())
        text_content = (
            f"Table {table}, column {column}: {meaning} "
            f"Stored codes: {codes_text}. Use the raw code in SQL, never the meaning."
        )
        yield f"{table}.{column} (glossary)", text_content


def rebuild(knowledge: Knowledge = None) -> int:
    """Embeds and inserts every documented fact into the Knowledge
    store. Meant to be run once, offline, via rebuild_index.py — or
    again whenever the JSON docs change. NOT on every app startup."""
    knowledge = knowledge or build_knowledge()
    count = 0
    for name, text_content in _iter_documents():
        knowledge.insert(name=name, text_content=text_content)
        count += 1
    logger.info("Rebuilt schema knowledge index: %d document(s) inserted", count)
    if count == 0:
        logger.warning(
            "No documentation found to index — table_descriptions.json, "
            "column_descriptions.json, and column_glossary.json are all "
            "empty or missing."
        )
    return count


_knowledge = None


def get_knowledge() -> Knowledge:
    global _knowledge
    if _knowledge is None:
        _knowledge = build_knowledge()
    return _knowledge


def search_schema_knowledge(query: str, top_k: int = 5) -> list[dict]:
    """Public entry point used by the agent tool (see glossary_tools.py).
    Note: Knowledge.search() takes the query as a positional string —
    num_documents/max_results is configured once on the Knowledge
    object's constructor (see build_knowledge above), not passed here."""
    knowledge = get_knowledge()
    results = knowledge.search(query)
    return [
        {
            "text": getattr(r, "content", None) or str(r),
            "name": getattr(r, "name", None),
        }
        for r in results[:top_k]
    ]