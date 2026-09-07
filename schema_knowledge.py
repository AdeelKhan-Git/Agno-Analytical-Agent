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


# ============================================================================
# nomic-embed-text is an ASYMMETRIC embedder: per its own model card, text
# meant to be SEARCHED FOR must be prefixed "search_query: " and text meant
# to be SEARCHED IN must be prefixed "search_document: " — retrieval quality
# drops measurably without this (confirmed independently by an external
# code review of this repo, and by Agno's own docs, which ship the exact
# same two-instance pattern for CohereEmbedder's input_type param —
# https://docs.agno.com/knowledge/concepts/embedder/cohere/cohere-embedder:
# "A single instance used by a vector database therefore applies the same
# input type to document insertion and query search" — document_embedder
# and query_embedder are built as two separate instances there).
# OllamaEmbedder has no built-in prefix/input_type concept at all (checked
# agno/knowledge/embedder/ollama.py — get_embedding() sends raw text
# unconditionally), so that split has to be implemented here.
# ============================================================================
class _PrefixedOllamaEmbedder(OllamaEmbedder):
    """Two instances of this (one per _prefix) are used below: one for
    indexing (rebuild.py — 'search_document: '), one for live query
    search (the running agent — 'search_query: '), both pointed at the
    SAME on-disk LanceDb table. The differing prefix only changes what
    text is fed to the model, not the resulting vector space or
    dimensionality — mixing them at index vs. query time is exactly how
    this model is meant to be used, not a mismatch.
    """
    _prefix: str = "search_document: "

    def get_embedding(self, text: str):
        return super().get_embedding(f"{self._prefix}{text}")

    async def async_get_embedding(self, text: str):
        return await super().async_get_embedding(f"{self._prefix}{text}")


class _DocumentEmbedder(_PrefixedOllamaEmbedder):
    _prefix = "search_document: "


class _QueryEmbedder(_PrefixedOllamaEmbedder):
    _prefix = "search_query: "


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_schema_knowledge(embedder) -> Knowledge:
    vector_db = LanceDb(
        uri=LANCEDB_URI,
        table_name=TABLE_NAME,
        search_type=SearchType.hybrid,
        embedder=embedder,
    )
    return Knowledge(vector_db=vector_db, max_results=20, contents_db=SqliteDb(db_file=CONTENTS_DB_FILE))


def build_knowledge() -> Knowledge:
    """Query-time construction — used by the live agent's
    search_knowledge_base tool. Uses 'search_query: ', matching how a
    user's question should be embedded, NOT how the indexed documents
    were embedded (see build_knowledge_for_indexing for that)."""
    return _build_schema_knowledge(_QueryEmbedder(id=EMBEDDER_MODEL, dimensions=768))


def build_knowledge_for_indexing() -> Knowledge:
    """Indexing-time construction — used ONLY by rebuild.py. Uses
    'search_document: ', matching how nomic-embed-text expects text that
    will be SEARCHED IN (as opposed to searched FOR) to be embedded."""
    return _build_schema_knowledge(_DocumentEmbedder(id=EMBEDDER_MODEL, dimensions=768))


def build_learning_knowledge() -> Knowledge:
    """The same asymmetric-prefix problem applies here in principle, but
    the clean two-instance split above does NOT apply cleanly: Agno's
    LearningMachine owns insert (save_learning) and search
    (search_learnings, and the automatic add_learnings_to_context recall
    before every run) internally, through this SAME Knowledge object,
    interleaved live during normal operation — there's no separate
    "index once, then only ever query" phase to split into two instances
    the way rebuild.py vs. the live agent works for the static schema
    docs above.
    Falls back to Nomic's own documented alternative for this exact
    situation: "If you want to do semantic similarity search instead of
    question answering, you should encode both queries and documents
    with the search_document task type" (Nomic Atlas docs). Using
    search_document uniformly on both sides is principled, not a
    shortcut — it's Nomic's own recommended fallback when query/document
    embedding calls can't be cleanly separated.
    """
    vector_db = LanceDb(
        uri="./learning/lancedb",
        table_name="learning_agent",
        search_type=SearchType.hybrid,
        embedder=_DocumentEmbedder(id=EMBEDDER_MODEL, dimensions=768),
    )
    return Knowledge(vector_db=vector_db, max_results=20, contents_db=SqliteDb(db_file=LEARNING_DB_FILE))


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

    # One document PER COLUMN, not one grouped document per table. This
    # keeps each column's embedding focused on just that column's meaning
    # (which column to use for which user intent), instead of diluting a
    # single vector across every column in the table. It costs more
    # documents/embedding calls, but retrieval precision is the actual
    # goal here — the model needs to reliably land on the ONE correct
    # column for a given question (e.g. jms_jcode vs subtitle vs title),
    # not get a vague "somewhere in this table" match.
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
    knowledge = knowledge or build_knowledge_for_indexing()
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