import json
import logging
import os
from collections import defaultdict
from agno.db.sqlite import SqliteDb
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
CONTENTS_DB_FILE = os.path.join(BASE_DIR, "knowledge_contents.db")
LEARNING_DB_FILE = os.path.join(BASE_DIR, "learning_knowledge_contents.db")

class NomicDocumentEmbedder(OllamaEmbedder):

    def get_embedding(self, text: str):
        return super().get_embedding(f"search_document: {text}")

    async def async_get_embedding(self, text: str):
        return await super().async_get_embedding(f"search_document: {text}")

    def get_embedding_and_usage(self, text: str):
        return super().get_embedding_and_usage(f"search_document: {text}")

    async def async_get_embedding_and_usage(self, text: str):
        return await super().async_get_embedding_and_usage(f"search_document: {text}")


class NomicQueryEmbedder(OllamaEmbedder):

    def get_embedding(self, text: str):
        return super().get_embedding(f"search_query: {text}")

    async def async_get_embedding(self, text: str):
        return await super().async_get_embedding(f"search_query: {text}")

    def get_embedding_and_usage(self, text: str):
        return super().get_embedding_and_usage(f"search_query: {text}")

    async def async_get_embedding_and_usage(self, text: str):
        return await super().async_get_embedding_and_usage(f"search_query: {text}")


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_table_docs_cache = None


def load_table_docs_for_retriever() -> dict:
    global _table_docs_cache
    if _table_docs_cache is not None:
        return _table_docs_cache

    table_descriptions = _load_json(TABLE_DESCRIPTIONS_PATH)
    join_partners = defaultdict(set)

    for table, entry in table_descriptions.items():
        for join in (entry or {}).get("joins", []):
            from_table = str(join.get("from", "")).split(".")[0]
            to_table = str(join.get("to", "")).split(".")[0]
            if from_table and to_table and from_table != to_table:
                join_partners[from_table].add(to_table)
                join_partners[to_table].add(from_table)

    _table_docs_cache = {
        "join_partners": {table: sorted(partners) for table, partners in join_partners.items()}
    }
    return _table_docs_cache


def build_knowledge(mode: str = "query") -> Knowledge:

    if mode == "index":
        embedder = NomicDocumentEmbedder(id=EMBEDDER_MODEL, dimensions=768)
    elif mode == "query":
        embedder = NomicQueryEmbedder(id=EMBEDDER_MODEL, dimensions=768)
    else:
        raise ValueError(f"build_knowledge(mode=...) must be 'index' or 'query', got {mode!r}")

    vector_db = LanceDb(
        uri=LANCEDB_URI,
        table_name=TABLE_NAME,
        search_type=SearchType.hybrid,
        embedder=embedder,
    )
    return Knowledge(vector_db=vector_db, max_results=10, contents_db=SqliteDb(db_file=CONTENTS_DB_FILE))


def build_learning_knowledge() -> Knowledge:
    embedder = NomicDocumentEmbedder(id=EMBEDDER_MODEL, dimensions=768)
    vector_db = LanceDb(
        uri="./learning/lancedb",
        table_name="learning_agent",
        search_type=SearchType.hybrid,
        embedder=embedder,
    )
    return Knowledge(vector_db=vector_db, max_results=5, contents_db=SqliteDb(db_file=LEARNING_DB_FILE))


def _iter_documents():
    table_descriptions = _load_json(TABLE_DESCRIPTIONS_PATH)
    column_descriptions = _load_json(DESCRIPTIONS_PATH)
    glossary = _load_json(GLOSSARY_PATH)

    # Table-level semantic anchors (also carries relationship/join guidance
    # inline, since almost every table description already documents its
    # joins — see the merged wording note further down in table_descriptions.json).
    for table, description in table_descriptions.items():
        yield (
            f"{table} (table)",
            f"TABLE: {table}. TABLE DESCRIPTION: {description}",
            {"table": table, "kind": "table"},
        )

    for table, columns in column_descriptions.items():
        for column, description in columns.items():
            yield (
                f"{table}.{column} (column)",
                f"TABLE: {table}. COLUMN: {column}. MEANING: {description}",
                {"table": table, "column": column, "kind": "column"},
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
            "When filtering this column, use the raw stored database code/value, not the human-readable meaning.",
            {"table": table, "column": column, "kind": "value_mapping"},
        )


def rebuild(knowledge: Knowledge = None) -> int:

    knowledge = knowledge or build_knowledge(mode="index")
    count = 0

    for name, text_content, metadata in _iter_documents():
        knowledge.insert(name=name, text_content=text_content, metadata=metadata)
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
        _knowledge = build_knowledge(mode="query")
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