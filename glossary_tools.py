import json
import logging
import os
import pandas as pd
from agno.tools import Toolkit
from db import ENGINE
import re
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

GLOSSARY_PATH = os.path.join(os.path.dirname(__file__), "column_glossary.json")


class GlossaryTools(Toolkit):

    def __init__(self):
        super().__init__(name="glossary_tools")
        self._glossary = self._load()
        self.register(self.get_column_codes)
        self.register(self.get_sample_values)
        self.register(self.list_documented_columns)
        self.register(self.resolve_filter_value)

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #

    def _load(self) -> dict:
        if not os.path.exists(GLOSSARY_PATH):
            return {}
        with open(GLOSSARY_PATH, "r") as f:
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

    # ------------------------------------------------------------------ #
    # tools exposed to the agent
    # ------------------------------------------------------------------ #

    def get_column_codes(self, table_name: str, column_name: str):
        key, mapping = self._lookup_key(table_name, column_name)
        if not mapping:
            logger.info("get_column_codes(%s, %s) -> no documented mapping", table_name, column_name)
            return (
                f"No documented mapping for {key}. Call get_sample_values to see "
                f"the actual stored values, and do not assume their meaning."
            )
        logger.info("get_column_codes(%s, %s) -> %s", table_name, column_name, mapping)
        return json.dumps(mapping)

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

    def list_documented_columns(self):
        logger.info("list_documented_columns() -> %d documented column(s)", len(self._glossary))
        return json.dumps(list(self._glossary.keys()))


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