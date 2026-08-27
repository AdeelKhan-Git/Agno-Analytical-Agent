import json
import logging
import os
import time
import pandas as pd
from sqlalchemy import inspect, text
from agno.tools import Toolkit
from db import ENGINE
import re
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

GLOSSARY_PATH = os.path.join(os.path.dirname(__file__), "column_glossary.json")
_CODE_LIKE_SQL_TYPES = ("char", "varchar", "text", "enum")
_FREE_TEXT_NAME_HINTS = (
    "desc", "description", "note", "notes", "bio", "biography", "policy",
    "guideline", "scope", "content", "text", "comment", "address", "url",
    "link", "email", "signature", "panel", "submenu", "banner",
)
_MAX_TEXT_LEN_FOR_CODE_LIKE = 60  # declared VARCHAR(N) longer than this is
                                  # treated as free text, not a short code


def _is_code_like_column(col: dict) -> bool:
    name_lower = col["name"].lower()
    type_str = str(col["type"]).lower()
    if not any(t in type_str for t in _CODE_LIKE_SQL_TYPES):
        return False
    if any(hint in name_lower for hint in _FREE_TEXT_NAME_HINTS):
        return False
    # try to read a declared length like VARCHAR(255) — if it's long,
    # treat as free text rather than a short code/identifier column
    m = re.search(r"\((\d+)\)", type_str)
    if m and int(m.group(1)) > _MAX_TEXT_LEN_FOR_CODE_LIKE:
        return False
    return True


class _CodeColumnIndex:
    """Discovers every code-like column across every table once, caches
    it in memory for a while, and can scan all of them for a value in a
    single pass. This is what replaces the old behavior of the LLM
    guessing which single column to check (title? subtitle? jms_jcode?)
    one at a time via repeated resolve_filter_value calls."""

    _REFRESH_SECONDS = 600  # rebuild the column list periodically in case
                             # the schema changes; cheap since it's just
                             # metadata reflection, not full table scans

    def __init__(self, engine):
        self._engine = engine
        self._columns: list[tuple[str, str]] = []  # (qualified_table, column_name)
        self._built_at = 0

    def _build(self):
        inspector = inspect(self._engine)
        try:
            schema_names = inspector.get_schema_names()
        except Exception:
            schema_names = [None]
        ignored = {
            "sys", "guest", "INFORMATION_SCHEMA", "information_schema",
            "performance_schema", "mysql",
        }
        schema_names = [s for s in schema_names if s not in ignored] or [None]

        columns = []
        for schema in schema_names:
            for table_name in inspector.get_table_names(schema=schema):
                qualified = f"{schema}.{table_name}" if schema else table_name
                try:
                    cols = inspector.get_columns(table_name, schema=schema)
                except Exception as e:
                    logger.debug("Skipping %s during code-column discovery: %s", qualified, e)
                    continue
                for c in cols:
                    if _is_code_like_column(c):
                        columns.append((qualified, c["name"]))

        self._columns = columns
        self._built_at = time.time()
        logger.info("Code-like column index built: %d column(s) across the schema", len(columns))

    def get_columns(self) -> list[tuple[str, str]]:
        if not self._columns or (time.time() - self._built_at) > self._REFRESH_SECONDS:
            self._build()
        return self._columns


_code_column_index = _CodeColumnIndex(ENGINE)


class GlossaryTools(Toolkit):

    def __init__(self):
        super().__init__(name="glossary_tools")
        self._glossary = self._load(GLOSSARY_PATH)
        self.register(self.get_sample_values)
        self.register(self.resolve_filter_value)
        self.register(self.find_value_anywhere)

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #

    def _load(self, path: str) -> dict:
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def _lookup_key(self, table_name: str, column_name: str):

        exact_key = f"{table_name}.{column_name}"
        if exact_key in self._glossary:
            return exact_key, self._glossary[exact_key]

        bare_table = table_name.split(".")[-1]
        target_suffix = f"{bare_table}.{column_name}"

        for glossary_key, mapping in self._glossary.items():
            glossary_bare = glossary_key.split(".")
            # last two segments of the glossary key = table.column
            if len(glossary_bare) >= 2 and ".".join(glossary_bare[-2:]) == target_suffix:
                return glossary_key, mapping

        return exact_key, None


    def get_sample_values(self, table_name: str, column_name: str):
        query = (
            f"SELECT DISTINCT {column_name} FROM {table_name} "
            f"WHERE {column_name} IS NOT NULL"
        )
        try:
            df = pd.read_sql_query(query, ENGINE)
            all_values = df[column_name].astype(str).tolist()
            logger.info(
                "get_sample_values(%s, %s) -> %d distinct value(s), truncated=%s",
                table_name, column_name, len(all_values), len(all_values) > 30,
            )
            return json.dumps({
                "values": all_values[:30],
                "truncated": len(all_values) > 30,
                "total_distinct_count": len(all_values),
            })
        except Exception as e:
            logger.warning("get_sample_values(%s, %s) failed: %s", table_name, column_name, e)
            return f"Could not sample values for {table_name}.{column_name}: {e}"


    def find_value_anywhere(self, search_term: str, table_name: str = None):
        """GENERAL-PURPOSE lookup for ANY short code, abbreviation, or
        identifier the user mentioned — a journal code, a role
        abbreviation, a status code, an editor's short ID, or anything
        else short and code-shaped. This is NOT specific to any one
        table or domain.

        Call this INSTEAD of repeatedly guessing single columns with
        resolve_filter_value when the user names a short term (a few
        characters, often uppercase or abbreviation-shaped, e.g. 'CMC',
        'EIC', 'RE') and you are not certain which column — or even
        which table — actually stores it. It scans every short-text/code
        column across the ENTIRE schema (or, if table_name is given,
        just that table) in one pass and returns every real match, with
        the table and column each was found in. This replaces trying
        title, then subtitle, then journal_subtitle, then giving up —
        instead it checks everywhere at once and tells you definitively
        where the term actually lives, or that it doesn't exist anywhere.

        Use this tool FIRST whenever the user's phrasing includes a
        short, code-shaped term and it is not immediately obvious which
        column holds it — before falling back to resolve_filter_value on
        individual guessed columns.
        """
        term = (search_term or "").strip()
        if not term:
            return json.dumps({"matches": [], "message": "search_term was empty."})

        term_lower = term.lower()
        columns = _code_column_index.get_columns()
        if table_name:
            bare = table_name.split(".")[-1].lower()
            columns = [(t, c) for t, c in columns if t.split(".")[-1].lower() == bare]

        matches = []
        checked = 0
        for qualified_table, column_name in columns:
            checked += 1
            try:
                query = text(
                    f"SELECT DISTINCT `{column_name}` FROM {qualified_table} "
                    f"WHERE `{column_name}` IS NOT NULL "
                    f"AND LOWER(`{column_name}`) LIKE :pattern LIMIT 20"
                )
                with ENGINE.connect() as conn:
                    rows = conn.execute(query, {"pattern": f"%{term_lower}%"}).fetchall()
            except Exception as e:
                logger.debug("find_value_anywhere: skipping %s.%s (%s)", qualified_table, column_name, e)
                continue

            for (value,) in rows:
                value_str = str(value)
                match_type = "exact" if value_str.lower() == term_lower else "partial"
                matches.append({
                    "table": qualified_table,
                    "column": column_name,
                    "value": value_str,
                    "match_type": match_type,
                })

        # exact matches first, then shorter partials (a shorter overlap
        # match is a stronger, less coincidental signal than a long one)
        matches.sort(key=lambda m: (0 if m["match_type"] == "exact" else 1, len(m["value"])))

        logger.info(
            "find_value_anywhere(%r, table_name=%s) -> checked %d column(s), %d match(es): %s",
            term, table_name, checked, len(matches), matches[:5],
        )

        if not matches:
            return json.dumps({
                "matches": [],
                "message": (
                    f"'{term}' was not found in any short code/identifier column "
                    f"checked ({checked} column(s) scanned{' on ' + table_name if table_name else ' across the schema'}). "
                    f"It may not exist, or it may only appear in a free-text column "
                    f"(like a title or description) — try resolve_filter_value on a "
                    f"specific free-text column if you have one in mind."
                ),
            })

        return json.dumps({"matches": matches[:15]})

    def resolve_filter_value(self,table_name: str,column_name: str,user_value: str):
        logger.info("resolve_filter_value(%s, %s, user_value=%r) called", table_name, column_name, user_value)

        query = f"""
        SELECT DISTINCT {column_name}
        FROM {table_name}
        WHERE {column_name} IS NOT NULL
        """

        try:
            df = pd.read_sql_query(query, ENGINE)
        except Exception as e:
            logger.warning("resolve_filter_value(%s, %s) query failed: %s", table_name, column_name, e)
            return json.dumps({
                "match_type": "error",
                "values": [],
                "message": str(e)
            })

        values = df[column_name].astype(str).tolist()
        # ---------------------------------------
        # Check glossary mappings FIRST
        # ---------------------------------------

        _, mapping = self._lookup_key(table_name, column_name)

        if mapping:
            user_norm = self._normalize(user_value)

            matches = []

            for stored_value, meaning in mapping.items():

                score = fuzz.token_set_ratio(
                    self._normalize(meaning),
                    user_norm
                )

                if score >= 80:
                    matches.append(stored_value)
                   

            if matches:
                logger.info("resolve_filter_value(%s, %s, %r) -> match_type=glossary, values=%s", table_name, column_name, user_value, matches)
                return json.dumps({
                    "match_type": "glossary",
                    "values": matches
                })
        # -----------------------
        # exact
        # -----------------------

        for v in values:
            if v.lower() == user_value.lower():
                logger.info("resolve_filter_value(%s, %s, %r) -> match_type=exact, value=%r", table_name, column_name, user_value, v)
                return json.dumps({
                    "match_type": "exact",
                    "values": [v]
                })

        # -----------------------
        # normalized
        # -----------------------

        normalized = self._normalize(user_value)

        matches = []

        for v in values:
            if self._normalize(v) == normalized:
                matches.append(v)

        if matches:
            logger.info("resolve_filter_value(%s, %s, %r) -> match_type=normalized, values=%s", table_name, column_name, user_value, matches)
            return json.dumps({
                "match_type": "normalized",
                "values": matches
            })

        # -----------------------
        # fuzzy
        # -----------------------

        scored = []

        for v in values:

            score = fuzz.token_set_ratio(
                normalized,
                self._normalize(v)
            )

            if score >= 80:
                scored.append((v, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        if scored:
            logger.info(
                "resolve_filter_value(%s, %s, %r) -> match_type=fuzzy, top_values=%s, scores=%s",
                table_name, column_name, user_value, [x[0] for x in scored][:5], [x[1] for x in scored][:5],
            )
            return json.dumps({

                "match_type": "fuzzy",

                "values": [x[0] for x in scored],

                "scores": [x[1] for x in scored]

            })

        logger.warning("resolve_filter_value(%s, %s, %r) -> match_type=none — no match found at all", table_name, column_name, user_value)
        return json.dumps({

            "match_type": "none",

            "values": []

        })
    def _normalize(self, text: str) -> str:
        text = text.lower()

        text = text.replace("&", "and")

        text = re.sub(r"[-_/]", " ", text)

        text = re.sub(r"[^\w\s]", "", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()