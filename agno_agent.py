import json,os
import logging
import re
import threading
import time
from typing import List, Literal, Optional, Callable
from pydantic import BaseModel, Field, field_validator, model_validator
from agno.models.openai import OpenAIChat
from agno.agent import Agent, Toolkit
from agno.models.google import Gemini
from agno.models.ollama import Ollama
from agno.models.vllm import VLLM
from agno.tools.sql import SQLTools
from agno.tools import tool
from agno.tools.pandas import PandasTools
from agno.learn import LearnedKnowledgeConfig, LearningMachine, LearningMode
from agno.run.agent import RunOutput, ToolCallStartedEvent, ToolCallCompletedEvent
from agno.db.sqlite import SqliteDb
from db import ENGINE, GEMINI_API_KEY, OPENAI_API_KEY, DB_URL
import pandas as pd
from instructions import INSTRUCTIONS
from schema_knowledge import (
    build_learning_knowledge,
    get_knowledge,
    load_table_docs_for_retriever,
    _format_doc_compact,
)

logger = logging.getLogger(__name__)

_SCHEMA_DOCS_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_TABLE_DESCRIPTIONS_PATH = os.path.join(_SCHEMA_DOCS_BASE_DIR, "table_descriptions.json")
_COLUMN_DESCRIPTIONS_PATH = os.path.join(_SCHEMA_DOCS_BASE_DIR, "column_descriptions.json")


def _load_json_doc(path: str) -> dict:
    if not os.path.exists(path):
        logger.warning("Schema doc file not found: %s", path)
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load/parse schema doc file: %s", path)
        return {}


def _format_curated_table_description(table_name: str, raw_columns_json: str) -> str:
    table_docs = _load_json_doc(_TABLE_DESCRIPTIONS_PATH)
    column_docs = _load_json_doc(_COLUMN_DESCRIPTIONS_PATH)

    table_entry = table_docs.get(table_name)
    documented_columns = column_docs.get(table_name, {})

    try:
        real_columns = json.loads(raw_columns_json)
    except Exception:
      
        return raw_columns_json

    if not isinstance(real_columns, list):
        return raw_columns_json

    lines = [f"TABLE {table_name}"]

    if table_entry and table_entry.get("description"):
        lines.append(str(table_entry["description"]))
    else:
        lines.append(
            "(No documented table-level description found for this table — "
            "treat column meanings below with extra care and verify with "
            "search_knowledge_base if the correct column is unclear.)"
        )

    documented_names = set(documented_columns.keys())
    real_column_names = [c.get("name") for c in real_columns if c.get("name")]
    real_column_meta = {c.get("name"): c for c in real_columns if c.get("name")}

    documented_lines = []
    undocumented_lines = []

    for col_name in real_column_names:
        meta = real_column_meta.get(col_name, {})
        col_type = meta.get("type", "?")
        if col_name in documented_names:
            meaning = documented_columns[col_name]
            documented_lines.append(f"  {col_name} ({col_type}): {meaning}")
        else:
            undocumented_lines.append(f"  {col_name} ({col_type}): undocumented — no confirmed meaning")

    if documented_lines:
        lines.append("")
        lines.append("COLUMNS")
        lines.extend(documented_lines)

    if undocumented_lines:
        lines.append("")
        lines.append(
            "RAW COLUMNS, UNDOCUMENTED — avoid using these unless the "
            "question specifically requires them and you have verified "
            "their meaning with search_knowledge_base or run_sql_query first:"
        )
        lines.extend(undocumented_lines)

    joins = (table_entry or {}).get("joins") or []
    if joins:
        lines.append("")
        lines.append("JOINS")
        for j in joins:
            on = j.get("on") or f"{j.get('from', '?')} = {j.get('to', '?')}"
            note = j.get("note", "")
            join_line = f"  {on}"
            if note:
                join_line += f"   -- {note}"
            lines.append(join_line)

    return "\n".join(lines)


class SchemaAwareSQLTools(Toolkit):
    def __init__(self, db_engine):
        super().__init__(name="schema_aware_sql_tools")

        self._inner = SQLTools(db_engine=db_engine)
        self.register(self.list_tables)
        self.register(self.describe_table)
        self.register(self.run_sql_query)

    @tool(cache_dir='agno_SQLTool_cache', cache_results=True, cache_ttl=3600)
    def list_tables(self) -> str:
        """List every table name available in the connected database.

        Call this first if you are not yet sure which tables exist.
        Returns a plain list of table names — it does not describe their
        columns; use describe_table(table_name) for that.
        """
        return self._inner.list_tables()

    @tool(cache_dir='agno_SQLTool_cache', cache_results=True, cache_ttl=3600)
    def describe_table(self, table_name: str) -> str:
        """Describe one table's real columns, types, documented meanings,
        and known joins to other tables.

        Args:
            table_name: The exact table name to describe, e.g. "journal".
                Must be a single table name string, not a list or a dict.

        Returns:
            A text block with the table description, each column's
            documented meaning where available, undocumented columns
            flagged as such, and any known JOIN relationships. Always
            call this for every table before writing SQL against it —
            never assume a column name or meaning.
        """
        table_name = table_name.strip()

        if not table_name:
            raise ValueError("table_name cannot be empty")

        raw_result = self._inner.describe_table(table_name)
        return _format_curated_table_description(table_name, raw_result)

    @tool(cache_dir='agno_SQLTool_cache', cache_results=True, cache_ttl=3600)
    def run_sql_query(self, query: str) -> str:
        """Execute a read-only SQL query against the database and return
        the results.

        Args:
            query: A single SELECT (or WITH ... SELECT) statement. Only
                use exact table/column names previously confirmed via
                describe_table — never guess a name.

        Returns:
            The query results. Use this to explore distinct values or
            verify assumptions before producing the final answer, not
            just to fetch the final answer's data.
        """
        return self._inner.run_sql_query(query)

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
        description=(
            "The SQL SELECT statement that answers the user's database question. "
            "This field is REQUIRED when is_refusal is false. "
            "Only leave it null when is_refusal is true."
        ),
    )

    chart_type: Literal["bar","pie","line","table","scatter","histogram",] = Field(
        default="table",
        description="Best-fit chart type for the SQL result.",
    )
    explanation: str = Field(
        ...,
        description=(
            "A short explanation of what the SQL query does. "
            "If is_refusal is true, explain why the request cannot be fulfilled."
        ),
    )
    is_refusal: bool = Field(
        default=False,
        description=(
            "Set to true only when the request cannot be fulfilled as a "
            "read-only SQL analysis. If false, sql_used must contain SQL."
        ),
    )
    insights: Optional[List[str]] = Field(
        default=None,
        description=(
            "Short analytical findings based only on the actual SQL results. "
            "Never invent numbers or findings."
        ),
    )

    @model_validator(mode="after")
    def _normalize_refusal(self):
        if self.is_refusal:
            self.sql_used = None
        return self

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
            return None

        normalized = []
        for item in value:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                text = (
                    item.get("message")
                    or item.get("text")
                    or item.get("insight")
                )
                normalized.append(text if text else str(item))
            else:
                normalized.append(str(item))
        return normalized



_RETRIEVER_TABLE_STAGE_RESULTS = 10
_RETRIEVER_COLUMN_STAGE_RESULTS_PER_TABLE = 15


def schema_knowledge_retriever(query: str, num_documents: int = None, **kwargs):

    # get_knowledge() returns a cached, mode="query" Knowledge instance —
    # correct task-prefix for embedding this search query, and reused
    # across calls instead of reopening LanceDb/the embedder every time.
    knowledge = get_knowledge()

    # Stage 1: which tables is this question actually about?
    table_docs = knowledge.search(
        query,
        max_results=_RETRIEVER_TABLE_STAGE_RESULTS,
        filters={"kind": "table"},
    )
    selected_tables = {
        doc.meta_data.get("table")
        for doc in table_docs
        if doc.meta_data and doc.meta_data.get("table")
    }

    if not selected_tables:
        # Nothing matched at the table level at all — fall back to an
        # unscoped search rather than returning nothing, so a genuinely
        # novel/oddly-worded question still gets SOME context.
        logger.warning(
            "schema_knowledge_retriever: no tables matched query %r in Stage 1, "
            "falling back to an unscoped search", query,
        )
        fallback_docs = knowledge.search(query, max_results=num_documents or 10)
        return [_format_doc_compact(doc) for doc in fallback_docs]

    # Union in each selected table's documented join partners, so a hub
    # table (e.g. tbl_ebm, which most cross-table questions need) is
    # never missing just because the question's wording pointed at the
    # tables on either side of it instead of the hub itself.
    join_partner_tables = load_table_docs_for_retriever().get("join_partners", {})
    expanded_tables = set(selected_tables)
    for table in selected_tables:
        expanded_tables.update(join_partner_tables.get(table, []))

    logger.info(
        "schema_knowledge_retriever: Stage 1 selected tables=%s, expanded with join partners=%s",
        sorted(selected_tables), sorted(expanded_tables - selected_tables),
    )

    # Stage 2: columns + value mappings, scoped to just those tables.
    # One filtered search per table (dict filters only — see note above
    # on why a single IN(...) call doesn't work on LanceDb), unioned and
    # de-duplicated by document name.
    seen_names = set()
    result_docs = list(table_docs)  # keep the table-level docs too — they carry join/description context
    for table in expanded_tables:
        table_scoped_docs = knowledge.search(
            query,
            max_results=_RETRIEVER_COLUMN_STAGE_RESULTS_PER_TABLE,
            filters={"table": table},
        )
        for doc in table_scoped_docs:
            if doc.name in seen_names:
                continue
            seen_names.add(doc.name)
            result_docs.append(doc)

    if num_documents is not None and len(result_docs) > num_documents:
        result_docs = result_docs[:num_documents]

    return [_format_doc_compact(doc) for doc in result_docs]


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

    logger.info("build_agent() model: %s", model_id)
    return Agent(
        model = Ollama(id = model_id, request_params={"think": False}, keep_alive=-1),
        # model=OpenAIChat(id="gpt-4o", api_key=OPENAI_API_KEY),
        # model=Gemini(id="gemini-3.5-flash-lite", api_key=GEMINI_API_KEY),

        # parser_model=Ollama(id=model_id, request_params={"think": False}, keep_alive=-1),
        tools=[
            SchemaAwareSQLTools(db_engine=ENGINE),
        ],
        knowledge=get_knowledge(),
        knowledge_retriever=schema_knowledge_retriever,
        learning=LearningMachine(
            knowledge=build_learning_knowledge(),
            learned_knowledge=LearnedKnowledgeConfig(mode=LearningMode.AGENTIC)
        ),
        search_knowledge=False,
        db=SqliteDb(db_file="agent_sessions.db"),
        add_knowledge_to_context=True,
        add_history_to_context=False,
        num_history_runs=1,
        description=(
            "You are a data analyst agent with direct access to a SQL database via your tools.\n"
        ),
        # compress_tool_results=True,
        output_schema=QueryPlan,
        retries=0,
        instructions=INSTRUCTIONS,
        markdown=False,
        debug_mode=1
    )

_agent = None
_agent_build_lock = threading.Lock()

_TOOL_PROGRESS_MESSAGES = {
    "list_tables": "📋 Looking at what tables are available...",
    "describe_table": "📐 Checking table structure...",
    "search_knowledge_base": "📚 Looking up documented business context...",
    "run_sql_query": "⚙️ Running SQL against the database...",
}
_DEFAULT_TOOL_PROGRESS = "🔧 Working on your question..."


def _get_agent():
    global _agent
    if _agent is None:
        with _agent_build_lock:
            if _agent is None:  # re-check inside the lock (double-checked locking)
                logger.info("Building agent instance (first call — will be reused for all subsequent requests)")
                _agent = build_agent()
    return _agent


def _dialect_name(db_url: str) -> str:
    db_url = db_url or ""
    for key in ("mssql", "mysql", "postgresql", "sqlite"):
        if db_url.startswith(key):
            return key
    return "unknown"


def _wrap_for_dry_run(sql: str, dialect: str) -> str:
    inner = sql.strip().rstrip(";")
    if dialect == "mysql" or dialect == "postgresql":
 
        return f"EXPLAIN {inner}"
    if dialect == "sqlite":
        return f"EXPLAIN QUERY PLAN {inner}"

    if dialect == "mssql":
        return f"SELECT TOP 1 1 AS _dry_run_ok FROM ({inner}) AS _dry_run_sq"
    return f"SELECT 1 AS _dry_run_ok FROM ({inner}) AS _dry_run_sq LIMIT 1"


def _validate_sql_executes(sql: str) -> list:
    dialect = _dialect_name(DB_URL)
    dry_run_sql = _wrap_for_dry_run(sql, dialect)
    started = time.perf_counter()
    try:

        with ENGINE.connect().execution_options(no_parameters=True) as conn:
            pd.read_sql_query(dry_run_sql, conn)
    except Exception as e:
        elapsed_ms = (time.perf_counter() - started) * 1000
        # Keep only the first line — the DB driver's message (e.g. "Unknown
        # column 'r.nicemat' in 'field list'") is what's actually useful;
        # the rest is SQLAlchemy/DBAPI traceback noise.
        db_message = str(e).strip().split("\n")[0]
        logger.warning(
            "SQL FAILED TO EXECUTE against the real database (%.1fms): %s | SQL: %s",
            elapsed_ms, db_message, sql,
        )
        return [
            f"SQL EXECUTION ERROR: the database rejected this SQL: {db_message}. "
            f"This means a table or column name in the SQL does not actually exist "
            f"(a hallucinated or misspelled name), or there is a syntax error. "
            f"Call describe_table() again for every table referenced in this SQL "
            f"and use ONLY the exact column names it returns — do not assume or "
            f"guess a column name, and do not confuse a column that belongs to one "
            f"joined table with a similarly-named one on another."
        ]
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info("SQL execution validation passed in %.1fms", elapsed_ms)
    if elapsed_ms > 2000:
        logger.warning(
            "SQL execution validation took %.1fms — unusually slow for an EXPLAIN/dry-run; "
            "check DB load or whether the dialect fallback (1-row subquery) is being used "
            "instead of a true EXPLAIN.",
            elapsed_ms,
        )
    return []


def _log_run_metrics(response: RunOutput) -> None:

    try:
        run_metrics = getattr(response, "metrics", None)
        if run_metrics is not None:
            logger.info("run metrics (aggregate): %r", run_metrics)

        messages = getattr(response, "messages", None) or []
        turn_count = 0
        for msg in messages:
            msg_metrics = getattr(msg, "metrics", None)
            if not msg_metrics:
                continue
            turn_count += 1
            role = getattr(msg, "role", "?")
            prompt_eval_count = getattr(msg_metrics, "input_tokens", None)
            output_tokens = getattr(msg_metrics, "output_tokens", None)
            # Ollama-specific timing fields, if the provider surfaces them
            # through Agno's Metrics object (prompt_eval_duration /
            # eval_duration in nanoseconds on the raw Ollama response).
            prompt_eval_duration = getattr(msg_metrics, "prompt_eval_duration", None)
            eval_duration = getattr(msg_metrics, "eval_duration", None)
            logger.info(
                "turn %d (%s) — input_tokens=%s output_tokens=%s "
                "prompt_eval_duration=%s eval_duration=%s",
                turn_count, role, prompt_eval_count, output_tokens,
                prompt_eval_duration, eval_duration,
            )
        logger.info("total turns with metrics this run: %d", turn_count)
    except Exception:
        # Metrics logging must never break an actual answer.
        logger.debug("Failed to log run metrics", exc_info=True)


def plan(user_prompt: str, _max_retries: int = 2, on_progress: Optional[Callable[[str], None]] = None,
         session_id: Optional[str] = None, _original_question: Optional[str] = None,
         on_step: Optional[Callable[[dict], None]] = None):

    if _original_question is None:
        _original_question = user_prompt

    logger.info("plan() called (retries left=%d, session_id=%s) — prompt: %r", _max_retries, session_id, user_prompt)
    agent = _get_agent()

    def _notify(message: str):
        if on_progress is None:
            return
        try:
            on_progress(message)
        except Exception:
            # A broken UI callback must never take down the actual query —
            # log and keep going.
            logger.debug("on_progress callback raised, ignoring", exc_info=True)

    def _emit_step(step: dict):
        if on_step is None:
            return
        try:
            on_step(step)
        except Exception:
            logger.debug("on_step callback raised, ignoring", exc_info=True)

    _notify("🔍 Understanding your question...")

    # Tracks in-flight calls by tool_call_id so the "completed" event can be
    # matched back to its "started" event to compute elapsed time and pair
    # the result with the original arguments in one combined UI step.
    _started_at: dict[str, float] = {}

    response = None
    for event in agent.run(user_prompt, stream=True, stream_events=True, yield_run_output=True,
                            session_id=session_id):
        if isinstance(event, ToolCallStartedEvent) and event.tool and event.tool.tool_name:
            tool_name = event.tool.tool_name
            tool_args = getattr(event.tool, "tool_args", None) or {}
            call_id = getattr(event.tool, "tool_call_id", None)
            if call_id:
                _started_at[call_id] = time.time()

            message = _TOOL_PROGRESS_MESSAGES.get(tool_name, _DEFAULT_TOOL_PROGRESS)
            logger.debug("Tool call started: %s -> %r", tool_name, message)
            _notify(message)
            _emit_step({
                "tool_name": tool_name,
                "tool_args": tool_args,
                "status": "started",
                "result": None,
                "elapsed_ms": None,
            })

        elif isinstance(event, ToolCallCompletedEvent) and event.tool and event.tool.tool_name:
            tool_name = event.tool.tool_name
            tool_args = getattr(event.tool, "tool_args", None) or {}
            call_id = getattr(event.tool, "tool_call_id", None)
            result = getattr(event.tool, "result", None)

            elapsed_ms = None
            if call_id and call_id in _started_at:
                elapsed_ms = (time.time() - _started_at.pop(call_id)) * 1000

            logger.debug("Tool call completed: %s (%.0fms)", tool_name,
                         elapsed_ms if elapsed_ms is not None else -1)
            _emit_step({
                "tool_name": tool_name,
                "tool_args": tool_args,
                "status": "completed",
                "result": result,
                "elapsed_ms": elapsed_ms,
            })

        elif isinstance(event, RunOutput):
            response = event

    if response is None:
        # Should not happen — yield_run_output=True guarantees a final
        # RunOutput is yielded — but fail loudly rather than silently
        # proceeding with an undefined `response`.
        raise RuntimeError("agent.run() streaming did not yield a final RunOutput")

    _log_run_metrics(response)

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

    # Single validation concern left: does this SQL actually execute against
    # the real database (real tables/columns, valid syntax)? Value-level
    # correctness (right filter value, right column choice) is now the
    # LearningMachine's job over time, not a per-request heuristic here.
    if result.sql_used and not result.is_refusal and _max_retries > 0:
        _validation_started = time.perf_counter()
        problems = _validate_sql_executes(result.sql_used)
        logger.info(
            "SQL validation took %.1fms (%d problem(s) found)",
            (time.perf_counter() - _validation_started) * 1000, len(problems),
        )
        if problems:
            logger.warning(
                "Validation found %d problem(s), triggering retry (retries left after this=%d): %s",
                len(problems), _max_retries - 1, problems,
            )
            _notify("🔁 Fixing an invalid table/column reference...")
            correction_prompt = (
                f"Your previous SQL was:\n{result.sql_used}\n\n"
                f"This SQL has problems:\n" + "\n".join(f"- {p}" for p in problems) +
                f"\n\nRe-answer the original question: {_original_question!r}\n"
                f"Do NOT set is_refusal — you must still produce a working sql_used."
            )
            retried = plan(correction_prompt, _max_retries=_max_retries - 1, on_progress=on_progress,
                            session_id=session_id, _original_question=_original_question, on_step=on_step)

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
                    "Could not produce SQL that executes successfully against the "
                    "database after retrying — please rephrase with more specific wording."
                )
            return result

    if not result.sql_used and not result.is_refusal and _max_retries > 0:
        logger.warning("No SQL produced and not a refusal — triggering retry to force real SQL")
        correction_prompt = (
            f"Re-answer the original question: {_original_question!r}\n"
            f"You did not produce any SQL last time and instead only wrote an explanation. "
            f"This question needs a real SQL query — write one (e.g. a SELECT DISTINCT for "
            f"a listing request) and put it in sql_used. Do not describe values from a "
            f"lookup tool in your explanation instead of running real SQL."
        )
        retried = plan(correction_prompt, _max_retries=_max_retries - 1, on_progress=on_progress,
                        session_id=session_id, _original_question=_original_question, on_step=on_step)
        if retried.sql_used or retried.is_refusal:
            return retried
        return result

    return result