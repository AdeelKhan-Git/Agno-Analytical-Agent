import json
import logging
import os
from collections import defaultdict

from agno.knowledge.knowledge import Knowledge
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType

logger = logging.getLogger(__name__)

LANCEDB_URI = "./data/lancedb"
EMBEDDER_MODEL = "nomic-embed-text"
TABLE_NAME = "schema_knowledge"

BASE_DIR = os.path.dirname(__file__)
GLOSSARY_PATH = os.path.join(BASE_DIR, "column_glossary.json")
DESCRIPTIONS_PATH = os.path.join(BASE_DIR, "column_descriptions.json")
TABLE_DESCRIPTIONS_PATH = os.path.join(BASE_DIR, "table_descriptions.json")


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_knowledge() -> Knowledge:
    vector_db = LanceDb(
        uri=LANCEDB_URI,
        table_name=TABLE_NAME,
        search_type=SearchType.hybrid,
        embedder=OllamaEmbedder(id=EMBEDDER_MODEL, dimensions=768),
    )
    return Knowledge(vector_db=vector_db, max_results=5)


def _table_columns(table: str, column_descriptions: dict) -> str:
    columns = column_descriptions.get(table, {})
    if not columns:
        return ""
    return "; ".join(f"{column}: {description}" for column, description in columns.items())


def _build_relationship_documents(table_descriptions: dict):
    for table, description in table_descriptions.items():
        text = str(description)
        lowered = text.lower()
        if "join" not in lowered and "connect" not in lowered and "relationship" not in lowered:
            continue

        yield (
            f"{table} (relationship guide)",
            f"Schema relationship guide for {table}. {description} "
            "Use the documented relationship path when this table is relevant. "
            "Do not invent alternate join keys or legacy identifiers."
        )


def _iter_documents():
    table_descriptions = _load_json(TABLE_DESCRIPTIONS_PATH)
    column_descriptions = _load_json(DESCRIPTIONS_PATH)
    glossary = _load_json(GLOSSARY_PATH)

    # Table-level semantic anchors.
    for table, description in table_descriptions.items():
        yield (
            f"{table} (table)",
            f"TABLE: {table}. TABLE DESCRIPTION: {description}"
        )

    #  Relationship-aware documents.
    yield from _build_relationship_documents(table_descriptions)

    # Group all columns for a table into one document. This prevents unrelated
    #    individual columns from consuming the top-k result budget.
    for table, columns in column_descriptions.items():
        column_text = _table_columns(table, column_descriptions)
        if column_text:
            yield (
                f"{table} (columns)",
                f"TABLE: {table}. COLUMNS AND DESCRIPTIONS: {column_text}"
            )

    # Keep coded-value mappings searchable by exact column/value terms.
    for qualified_col, mapping in glossary.items():
        parts = qualified_col.rsplit(".", 1)
        if len(parts) != 2:
            continue
        table, column = parts
        meaning = mapping.get("meaning", "")
        codes = {k: v for k, v in mapping.items() if k != "meaning"}
        codes_text = "; ".join(f"{k} = {v}" for k, v in codes.items())
        yield (
            f"{table}.{column} (value mapping)",
            f"TABLE: {table}. COLUMN: {column}. "
            f"MEANING: {meaning}. STORED VALUES/CODES: {codes_text}. "
            "When filtering this column, use the raw stored database code/value, not the human-readable meaning."
        )


def rebuild(knowledge: Knowledge = None) -> int:

    knowledge = knowledge or build_knowledge()
    count = 0

    for name, text_content in _iter_documents():
        knowledge.insert(name=name, text_content=text_content)
        count += 1

    logger.info("Rebuilt schema knowledge index: %d document(s) inserted", count)
    if count == 0:
        logger.warning(
            "No documentation found to index — table_descriptions.json, "
            "column_descriptions.json, and column_glossary.json are empty or missing."
        )
    return count


_knowledge = None


def get_knowledge() -> Knowledge:
    global _knowledge
    if _knowledge is None:
        _knowledge = build_knowledge()
    return _knowledge


def search_schema_knowledge(query: str, top_k: int = 5) -> list[dict]:
    knowledge = get_knowledge()
    results = knowledge.search(query)
    return [
        {
            "text": getattr(r, "content", None) or str(r),
            "name": getattr(r, "name", None),
        }
        for r in results[:top_k]
    ]
