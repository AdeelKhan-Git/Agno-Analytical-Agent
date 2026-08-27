import json,os
import logging
import re
import threading
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from agno.models.openai import OpenAIChat
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.ollama import Ollama
from agno.tools.sql import SQLTools
from agno.tools.pandas import PandasTools
from glossary_tools import GlossaryTools
from db import ENGINE, GEMINI_API_KEY, OPENAI_API_KEY
import pandas as pd
from instructions import INSTRUCTIONS
from schema_knowledge import build_knowledge

logger = logging.getLogger(__name__)


CHART_TYPE_ALIASES = {
    "scatterplot": "scatter",
    "scatter_plot": "scatter",
    "histogram": "histogram",
    "hist": "histogram",
    "boxplot": "table",
    "box": "table",
    "heatmap": "table",
    "area": "line",
    "column": "bar",
    "barchart": "bar",
    "piechart": "pie",
    "linechart": "line",
}

VALID_CHART_TYPES = ("bar", "pie", "line", "table", "scatter", "histogram")

class QueryPlan(BaseModel):
    sql_used: Optional[str] = Field(
        default=None,
        description="The final, validated SQL SELECT statement that answers the question. "
        "Leave this as null/omitted if is_refusal is true — do not provide a SELECT at all "
        "in that case.",
    )
    chart_type: Literal["bar", "pie", "line", "table", "scatter", "histogram"] = Field(
        default="table",
        description="Best-fit chart type for the shape of this result. Irrelevant when "
        "is_refusal is true — just leave the default.",
    )
    explanation: str = Field(
        ..., description="One or two sentences on what the query does, OR — if is_refusal is "
        "true — a clear, direct message saying you cannot fulfill the request and why."
    )
    is_refusal: bool = Field(
        default=False,
        description="Set to true ONLY when the request cannot be fulfilled at all — e.g. it "
        "requires a write operation (update/insert/delete), or asks for something outside a "
        "read-only data analyst's ability. When true, sql_used must be omitted/null, and "
        "explanation must clearly state you cannot do this and why.",
    )
    insights: Optional[List[str]] = Field(
        default=None,
        description="Optional list of short, concrete analytical findings (trends, outliers, "
        "notable comparisons) if the question asked for analysis rather than just a chart. "
        "Base these only on aggregate computations you actually ran — never invent numbers.",
    )

    @field_validator("chart_type", mode="before")
    @classmethod
    def _normalize_chart_type(cls, value):
        if not isinstance(value, str):
            return "table"
        lowered = value.strip().lower()
        if lowered in VALID_CHART_TYPES:
            return lowered
        return CHART_TYPE_ALIASES.get(lowered, "table")

    @field_validator("insights", mode="before")
    @classmethod
    def _normalize_insights(cls, value):
        if value is None:
            return value
        normalized = []
        for item in value:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                text = item.get("message") or item.get("text") or item.get("insight")
                normalized.append(text if text else str(item))
            else:
                normalized.append(str(item))
        return normalized


def build_agent():
    model_id = os.environ.get("OLLAMA_MODEL")
    if not model_id:
        raise RuntimeError(
            "OLLAMA_MODEL is not set. This must be exported by entrypoint.sh "
            "after it builds the tuned model variant — running this app without "
            "that means num_ctx/num_predict/temperature are not correctly "
            "configured. If you're running locally without entrypoint.sh, set "
            "OLLAMA_MODEL explicitly to a model you've already tuned/pulled."
        )
    logger.info("build_agent() using Ollama model: %s", model_id)
    return Agent(
        model =  Ollama(id=model_id),
        # model=OpenAIChat(id="gpt-4o", api_key=OPENAI_API_KEY),
        # model=Gemini(id="gemini-2.5-flash", api_key=GEMINI_API_KEY),
        tools=[
            SQLTools(db_engine=ENGINE),
            PandasTools(),
            GlossaryTools(),
        ],
        knowledge=build_knowledge(),
        search_knowledge=True,
        description=( 
            "You are a data analyst agent with direct access to a SQL database via your tools.\n"
             ),
        output_schema=QueryPlan,
        use_json_mode=True,
        instructions=INSTRUCTIONS,
        markdown=False,
    )

_agent = None
_agent_lock = threading.Lock()


def _get_agent():
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:  # re-check inside the lock (double-checked locking)
                logger.info("Building agent instance (first call — will be reused for all subsequent requests)")
                _agent = build_agent()
    return _agent


# ============================================================================
# SECTION: SQL literal-value validator — catches hallucinated/ambiguous
# filter values that the agent's own SQL used, before we trust it
# ============================================================================
def _extract_literal_filters(sql: str):
    filters = []
    eq_pattern = r"(\w+(?:\.\w+)?)\s*=\s*'([^']*)'"
    for col, val in re.findall(eq_pattern, sql):
        filters.append((col.split(".")[-1], [val]))

    in_pattern = r"(\w+(?:\.\w+)?)\s+IN\s*\(([^)]*)\)"
    for col, values_blob in re.findall(in_pattern, sql, flags=re.IGNORECASE):
        values = re.findall(r"'([^']*)'", values_blob)
        if values:
            filters.append((col.split(".")[-1], values))

    return filters


def _extract_main_table(sql: str):
    match = re.search(r"FROM\s+([\w\.]+)", sql, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _validate_literals(sql: str):
    table = _extract_main_table(sql)
    if not table:
        return []

    problems = []
    for column, values in _extract_literal_filters(sql):
        try:
            query = (
                f"SELECT DISTINCT {column} FROM {table} "
                f"WHERE {column} IS NOT NULL"
            )
            df = pd.read_sql_query(query, ENGINE)
            actual_values_raw = df[column].astype(str).tolist()
            actual_values_lower = {v.lower() for v in actual_values_raw}
        except Exception as e:
            logger.debug("_validate_literals could not sample %s.%s: %s", table, column, e)
            continue

        for v in values:
            v_lower = v.lower()

            if v_lower not in actual_values_lower:
                sample = actual_values_raw[:10]
                logger.warning(
                    "HALLUCINATED value detected: %s.%s = %r does not exist. Real values include: %s",
                    table, column, v, sample,
                )
                problems.append(
                    f"HALLUCINATED: column '{column}' does not actually contain the value "
                    f"'{v}'. Real stored values include: {sample}"
                )
                continue

            similar_others = [
                other for other in actual_values_raw
                if other.lower() != v_lower
                and (v_lower in other.lower() or other.lower() in v_lower)
            ]
            if similar_others:
                logger.warning(
                    "AMBIGUOUS value detected: %s.%s = %r has similar-but-different values: %s",
                    table, column, v, similar_others,
                )
                problems.append(
                    f"AMBIGUOUS: column '{column}' was filtered to '{v}', but these other "
                    f"DIFFERENT stored values are similar and may also be relevant to the "
                    f"user's question: {similar_others}. Resolve this by using an exact match "
                    f"if the user's wording matches one of them exactly, or by including ALL "
                    f"relevant values with IN (...) — do not refuse the question over this."
                )
    return problems


# ============================================================================
# SECTION: dead/empty-column validator — catches the model selecting a
# column that is real (not hallucinated) but effectively always NULL in
# production, so the query "succeeds" while silently returning nothing
# useful. General on purpose: works for ANY table/column, computed live
# from actual data, not a hardcoded list of known-dead columns.
# ============================================================================
_DEAD_COLUMN_NULL_RATIO_THRESHOLD = 0.98  # >=98% NULL across the sample counts as dead
_DEAD_COLUMN_SAMPLE_ROWS = 500


def _extract_select_columns(sql: str) -> list[tuple[str, str]]:
    """Returns (table_hint_or_None, column_name) pairs for columns that
    appear in the SELECT list specifically — not WHERE/JOIN/GROUP BY,
    since a dead column being FILTERED on (e.g. checking it's non-null)
    is a different, less risky pattern than a dead column being the
    actual thing returned to the user."""
    match = re.search(r"SELECT\s+(.*?)\s+FROM\s+", sql, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    select_list = match.group(1)
    if select_list.strip() == "*":
        return []

    columns = []
    for raw in select_list.split(","):
        raw = raw.strip()
        # strip aggregate wrappers like COUNT(x), SUM(x) — those aren't
        # "selecting the raw column value" the way a bare column ref is
        if re.match(r"^\w+\s*\(", raw):
            continue
        # drop an "AS alias" suffix
        raw = re.split(r"\s+AS\s+", raw, flags=re.IGNORECASE)[0].strip()
        m = re.match(r"^(?:(\w+)\.)?(\w+)$", raw)
        if m:
            table_hint, col = m.groups()
            columns.append((table_hint, col))
    return columns


def _validate_selected_columns_not_dead(sql: str) -> list:
    table = _extract_main_table(sql)
    if not table:
        return []

    select_cols = _extract_select_columns(sql)
    if not select_cols:
        return []

    problems = []
    for _table_hint, column in select_cols:
        try:
            query = (
                f"SELECT "
                f"SUM(CASE WHEN {column} IS NULL OR {column} = '' THEN 1 ELSE 0 END) AS null_count, "
                f"COUNT(*) AS total_count "
                f"FROM (SELECT {column} FROM {table} LIMIT {_DEAD_COLUMN_SAMPLE_ROWS}) AS sample"
            )
            df = pd.read_sql_query(query, ENGINE)
            null_count = int(df["null_count"].iloc[0] or 0)
            total_count = int(df["total_count"].iloc[0] or 0)
        except Exception as e:
            logger.debug("_validate_selected_columns_not_dead could not sample %s.%s: %s", table, column, e)
            continue

        if total_count == 0:
            continue
        null_ratio = null_count / total_count
        if null_ratio >= _DEAD_COLUMN_NULL_RATIO_THRESHOLD:
            logger.warning(
                "DEAD COLUMN selected: %s.%s is %.0f%% NULL/empty across a %d-row sample",
                table, column, null_ratio * 100, total_count,
            )
            problems.append(
                f"DEAD COLUMN: '{column}' on {table} is {null_ratio * 100:.0f}% NULL/empty "
                f"across a real sample of {total_count} rows — selecting it will not return "
                f"a useful answer even though the column exists and the query runs "
                f"successfully. Call get_column_description('{table}', '{column}') and/or "
                f"get_table_description('{table}') to find where this data actually lives "
                f"(often a different, similarly-named column, or a join to another table) "
                f"before finalizing your SQL."
            )
    return problems


def plan(user_prompt: str, _max_retries: int = 2):
    logger.info("plan() called (retries left=%d) — prompt: %r", _max_retries, user_prompt)
    agent = _get_agent()
    with _agent_lock:
        response = agent.run(user_prompt)
    content = response.content

    result = None
    if isinstance(content, QueryPlan):
        result = content
    elif isinstance(content, str):
        logger.warning("Model returned a raw string instead of structured output — attempting JSON parse")
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            data = json.loads(cleaned)
            result = QueryPlan(**data)
        except Exception:
            logger.error("Failed to parse model output as QueryPlan. Raw response: %r", content)
            raise RuntimeError(
                f"Agent did not return valid structured output. Raw response: {content!r}"
            )
    else:
        logger.error("Unexpected response content type: %s", type(content))
        raise RuntimeError(f"Unexpected response content type: {type(content)}")

    logger.info(
        "plan() result — is_refusal=%s, sql_used=%r, chart_type=%s",
        result.is_refusal, result.sql_used, result.chart_type,
    )

    if result.sql_used and not result.is_refusal and _max_retries > 0:
        problems = _validate_literals(result.sql_used)
        problems += _validate_selected_columns_not_dead(result.sql_used)
        if problems:
            logger.warning(
                "Validation found %d problem(s), triggering retry (retries left after this=%d): %s",
                len(problems), _max_retries - 1, problems,
            )
            correction_prompt = (
                f"Your previous SQL was:\n{result.sql_used}\n\n"
                f"This SQL has problems:\n" + "\n".join(f"- {p}" for p in problems) +
                f"\n\nRe-answer the original question: {user_prompt!r}\n"
                f"Use resolve_filter_value(table_name, column_name, user_value) to find the "
                f"correct real stored value(s) before writing new SQL — it checks the FULL set "
                f"of distinct values, not a capped sample, so it will find the right value even "
                f"in high-cardinality columns. Use exactly what it returns. Do NOT set "
                f"is_refusal — you must still produce a working sql_used."
            )
            retried = plan(correction_prompt, _max_retries=_max_retries - 1)

            if retried.sql_used:
                logger.info("Retry produced usable SQL — using retried result")
                return retried
            if result.sql_used:
                logger.info("Retry produced nothing usable — falling back to original (imperfect) result")
                return result
            logger.warning("Neither original nor retry produced usable SQL — marking as refusal")
            result.is_refusal = True
            if not result.explanation:
                result.explanation = (
                    "Could not confidently resolve ambiguous filter values for this "
                    "question after retrying — please rephrase with more specific wording."
                )
            return result


    if not result.sql_used and not result.is_refusal and _max_retries > 0:
        logger.warning("No SQL produced and not a refusal — triggering retry to force real SQL")
        correction_prompt = (
            f"Re-answer the original question: {user_prompt!r}\n"
            f"You did not produce any SQL last time and instead only wrote an explanation. "
            f"This question needs a real SQL query — write one (e.g. a SELECT DISTINCT for "
            f"a listing request) and put it in sql_used. Do not describe values from a "
            f"lookup tool in your explanation instead of running real SQL."
        )
        retried = plan(correction_prompt, _max_retries=_max_retries - 1)
        if retried.sql_used or retried.is_refusal:
            return retried
        return result

    return result