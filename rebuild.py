"""
Builds (or rebuilds) the semantic search index — using Agno's own
Knowledge + OllamaEmbedder + LanceDb (see schema_knowledge.py) — from
table_descriptions.json, column_descriptions.json, and
column_glossary.json.

Run this:
  - once, before first use
  - again, any time you edit any of those three JSON files

Do NOT run this on every app/container start — it embeds every
document via Ollama, which is unnecessary cost on a restart when the
index already exists on a persisted volume.

Usage:
    python rebuild_index.py
"""
import logging

from schema_knowledge import rebuild

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")


def main():
    count = rebuild()
    print(f"Indexed {count} document(s).")


if __name__ == "__main__":
    main()