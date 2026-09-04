import logging
from logging_config import setup_logging
setup_logging()

import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

from agno_agent import plan
from db import run_sql, DB_URL

logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Analytics Platform", page_icon="📊", layout="wide")

# ============================================================================
# THEME / STYLING
# ============================================================================
PALETTE = px.colors.qualitative.Set2  # consistent, professional color sequence
ACCENT = "#4F46E5"  # indigo — used for buttons, links, highlights

st.markdown(f"""
<style>
.block-container {{ padding-top: 1.5rem; max-width: 1400px; }}

/* Header banner */
.app-header {{
    padding: 20px 28px;
    border-radius: 16px;
    background: linear-gradient(135deg, {ACCENT} 0%, #7C3AED 100%);
    color: white;
    margin-bottom: 20px;
}}
.app-header h1 {{ margin: 0; font-size: 1.6rem; }}
.app-header p {{ margin: 4px 0 0 0; opacity: 0.9; font-size: 0.95rem; }}

/* Buttons */
.stButton>button {{
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid #e5e7eb;
    transition: all 0.15s ease;
}}
.stButton>button:hover {{
    border-color: {ACCENT};
    color: {ACCENT};
}}

/* Insight cards */
.card {{
    padding: 14px 18px;
    border-left: 3px solid {ACCENT};
    border-radius: 10px;
    background: #f8f8fc;
    margin-bottom: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}

/* Example prompt chips */
.chip-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}

/* Metric cards */
[data-testid="stMetric"] {{
    background: #fafafa;
    border: 1px solid #eee;
    border-radius: 12px;
    padding: 10px 14px;
}}

/* Tabs */
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}

/* Sidebar connection badge */
.conn-badge {{
    display: flex; align-items: center; gap: 8px;
    padding: 10px 14px; border-radius: 10px;
    background: #f0fdf4; border: 1px solid #bbf7d0;
    font-size: 0.85rem; margin-bottom: 14px;
}}
.conn-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: #22c55e; display: inline-block;
}}

/* Sidebar stat pills */
.stat-pill {{
    background: #fafafa; border: 1px solid #eee; border-radius: 10px;
    padding: 8px 12px; text-align: center;
}}
.stat-pill .num {{ font-size: 1.3rem; font-weight: 700; color: {ACCENT}; }}
.stat-pill .lbl {{ font-size: 0.7rem; color: #888; text-transform: uppercase; }}

/* History entry chart-type badge */
.chart-badge {{
    display: inline-block; font-size: 0.65rem; padding: 1px 6px;
    border-radius: 6px; background: #eef2ff; color: {ACCENT};
    font-weight: 600; margin-right: 4px;
}}

/* ======================================================================
   FULL-SCREEN LOADER — the ONLY thing visible while a request is in
   flight. One surface, one source of truth: no st.status() running
   alongside it (that was the earlier bug — two competing UI systems,
   one hidden behind the other). Content here is deliberately limited to
   clean, human-readable step labels — never raw SQL, tool args, or JSON.
   Full technical detail is still available afterward in the SQL tab.
   ====================================================================== */
.loading-overlay {{
    position: fixed;
    inset: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(3px);
}}
.loading-card {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 22px;
    padding: 40px 48px;
    border-radius: 20px;
    background: white;
    box-shadow: 0 20px 60px rgba(79, 70, 229, 0.12), 0 2px 8px rgba(0,0,0,0.06);
    width: min(480px, 90vw);
    animation: card-in 0.25s ease-out;
}}
@keyframes card-in {{
    from {{ opacity: 0; transform: translateY(8px) scale(0.98); }}
    to   {{ opacity: 1; transform: translateY(0) scale(1); }}
}}
.loading-spinner-big {{
    width: 46px;
    height: 46px;
    border: 4px solid #eef2ff;
    border-top: 4px solid {ACCENT};
    border-radius: 50%;
    animation: loading-spin 0.8s linear infinite;
}}
@keyframes loading-spin {{
    0% {{ transform: rotate(0deg); }}
    100% {{ transform: rotate(360deg); }}
}}
.loading-headline {{
    font-weight: 700;
    color: #1f2937;
    font-size: 1.05rem;
    text-align: center;
}}
.loading-steps {{
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 2px;
}}
.loading-step {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 4px;
    font-size: 0.88rem;
    color: #9ca3af;
    transition: color 0.2s ease;
}}
.loading-step.is-done {{
    color: #374151;
}}
.loading-step.is-active {{
    color: {ACCENT};
    font-weight: 600;
}}
.loading-step-icon {{
    width: 20px;
    text-align: center;
    flex-shrink: 0;
    font-size: 0.95rem;
}}
.loading-step.is-active .loading-step-icon {{
    animation: pulse-icon 1s ease-in-out infinite;
}}
@keyframes pulse-icon {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.4; }}
}}

/* ======================================================================
   LEARNING-MODE LOADER — a visually distinct variant of the loading
   overlay shown ONLY when the user's message is an "error:" correction.
   Amber/orange instead of indigo, and a different headline/icon set, so
   it's immediately clear this is a different kind of request (teaching
   the agent a correction) rather than a normal question being answered.
   Reuses the exact same card/step/spinner structure as the regular
   loader — only the color and copy differ.
   ====================================================================== */
.learning-card {{
    box-shadow: 0 20px 60px rgba(217, 119, 6, 0.14), 0 2px 8px rgba(0,0,0,0.06);
}}
.learning-spinner-big {{
    width: 46px;
    height: 46px;
    border: 4px solid #fef3c7;
    border-top: 4px solid #d97706;
    border-radius: 50%;
    animation: loading-spin 0.8s linear infinite;
}}
.learning-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    background: #fef3c7;
    color: #92400e;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-bottom: 4px;
}}
.learning-step.is-done {{ color: #374151; }}
.learning-step.is-active {{ color: #d97706; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# STATE
# ============================================================================
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"prompt", "chart_type", "rows", "time"}
if "plan" not in st.session_state:
    st.session_state.plan = None
    st.session_state.df = None
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = None
if "current_is_learning" not in st.session_state:
    st.session_state.current_is_learning = False

# agent_session_id: one stable UUID per BROWSER session, generated once and
# kept in st.session_state so it survives reruns within the same browser
# tab but is unique per user/tab. Passed to plan() -> agent.run(session_id=...)
# so Agno's own conversation history is scoped correctly per user — without
# this, every prompt looks like a brand new, unrelated conversation to Agno,
# and the model has no memory of already having called describe_table on a
# table earlier in the SAME user's session.
if "agent_session_id" not in st.session_state:
    import uuid
    st.session_state.agent_session_id = str(uuid.uuid4())


def _dialect_label(db_url: str) -> str:
    if not db_url:
        return "Not configured"
    mapping = {
        "mssql": "SQL Server", "postgresql": "PostgreSQL",
        "mysql": "MySQL", "sqlite": "SQLite",
    }
    for prefix, label in mapping.items():
        if db_url.startswith(prefix):
            return label
    return "Unknown"


def _db_name(db_url: str) -> str:
    """Best-effort extraction of the database name from a connection string,
    for display only — never used for anything security-sensitive."""
    if not db_url:
        return "—"
    try:
        return db_url.rsplit("/", 1)[-1].split("?")[0] or "—"
    except Exception:
        return "—"


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("📊 Analytics")

    # --- Connection status ---
    st.markdown(f"""
    <div class="conn-badge">
        <span class="conn-dot"></span>
        <div>
            <div style="font-weight:600;">{_dialect_label(DB_URL)}</div>
            <div style="color:#666;">{_db_name(DB_URL)} · read-only</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Session stats ---
    total_queries = len(st.session_state.history)
    total_rows = sum(h.get("rows", 0) for h in st.session_state.history)
    s1, s2 = st.columns(2)
    s1.markdown(f'<div class="stat-pill"><div class="num">{total_queries}</div><div class="lbl">Queries</div></div>', unsafe_allow_html=True)
    s2.markdown(f'<div class="stat-pill"><div class="num">{total_rows:,}</div><div class="lbl">Rows fetched</div></div>', unsafe_allow_html=True)

    st.divider()

    # --- Settings ---
    with st.expander("⚙️ Settings", expanded=False):
        show_sql = st.checkbox("Show SQL", True)
        show_insights = st.checkbox("Show Insights", True)

    # --- Actions ---
    b1, b2 = st.columns(2)
    if b1.button("🆕 New", width='stretch', help="Clear the current result, keep history"):
        st.session_state.plan = None
        st.session_state.df = None
        st.session_state.current_prompt = None
        st.session_state.current_is_learning = False
        st.rerun()
    if b2.button("🗑 Reset", width='stretch', help="Clear everything including history"):
        st.session_state.history = []
        st.session_state.plan = None
        st.session_state.df = None
        st.session_state.current_prompt = None
        st.session_state.current_is_learning = False
        st.rerun()

    # --- History ---
    if st.session_state.history:
        st.divider()
        st.subheader("History")
        recent = list(reversed(st.session_state.history[-10:]))
        for idx, h in enumerate(recent):
            label = h["prompt"]
            short = label[:32] + ("…" if len(label) > 32 else "")
            col_badge, col_btn, col_refresh = st.columns([1, 3, 1])
            col_badge.markdown(f'<span class="chart-badge">{h.get("chart_type","?")[:3].upper()}</span>', unsafe_allow_html=True)

            # Clicking the label restores the CACHED result instantly —
            # no new LLM call, no new DB query.
            if col_btn.button(short, key=f"hist_{idx}", width='stretch', help="Show cached result instantly"):
                st.session_state.plan = h["query_plan"]
                st.session_state.df = h["dataframe"]
                st.session_state.current_prompt = h["prompt"]  # full text, for the main-page display
                st.session_state.current_is_learning = h.get("is_learning", False)
                st.rerun()

            # Small refresh icon to deliberately re-run against the DB,
            # in case the underlying data has changed since this was cached.
            if col_refresh.button("🔄", key=f"refresh_{idx}", help="Re-run fresh (ignores cache)"):
                st.session_state["force_prompt"] = label
                st.rerun()
    else:
        st.caption("No queries yet this session.")

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="app-header">
    <h1>📊 AI Analytics Platform</h1>
    <p>Ask questions in plain English — get validated SQL, real data, and the right chart.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# COLUMN INTELLIGENCE — auto-detect what each column actually represents,
# so chart defaults are sensible instead of just "first column available".
# ============================================================================
ID_HINTS = ("_id", "id")
DATE_HINTS = ("date", "_at", "time", "timestamp", "created", "updated")
MONEY_HINTS = ("price", "total", "amount", "spent", "revenue", "cost", "sales",
               "value", "earning", "balance", "payment")
COUNT_HINTS = ("count", "qty", "quantity", "sold", "rank", "num_", "total_sold")
NAME_HINTS = ("name", "category", "brand", "store", "city", "status", "type",
              "region", "segment", "province", "occupation")


def _matches_any(col_lower: str, hints: tuple) -> bool:
    return any(h in col_lower for h in hints)


def analyze_columns(df: pd.DataFrame) -> dict:
    """Classifies each column by likely meaning, using name heuristics plus
    actual dtype/cardinality — so chart defaults pick a sensible metric
    column instead of an ID, and a sensible category instead of a raw key."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    id_cols, date_cols, money_cols, count_cols, name_cols = [], [], [], [], []

    for col in all_cols:
        lowered = col.lower()
        is_id = lowered == "id" or lowered.endswith("_id")
        is_high_cardinality_text = (
            df[col].dtype == object and df[col].nunique(dropna=True) >= max(1, len(df) * 0.9)
        )
        if is_id or is_high_cardinality_text:
            id_cols.append(col)

        if pd.api.types.is_datetime64_any_dtype(df[col]) or _matches_any(lowered, DATE_HINTS):
            date_cols.append(col)
        if _matches_any(lowered, MONEY_HINTS):
            money_cols.append(col)
        if _matches_any(lowered, COUNT_HINTS):
            count_cols.append(col)
        if _matches_any(lowered, NAME_HINTS):
            name_cols.append(col)

    numeric_non_id = [c for c in numeric_cols if c not in id_cols]
    categorical_non_id = [c for c in all_cols if c not in numeric_cols and c not in id_cols]

    return {
        "id_cols": id_cols,
        "date_cols": date_cols,
        "money_cols": [c for c in money_cols if c in numeric_cols],
        "count_cols": [c for c in count_cols if c in numeric_cols],
        "name_cols": [c for c in name_cols if c not in id_cols],
        "numeric_non_id": numeric_non_id,
        "categorical_non_id": categorical_non_id,
    }


def _first_available(*lists, fallback=None):
    """
    Returns the first available column from the supplied lists.
    Never raises IndexError.
    """
    for lst in lists:
        if lst:
            return lst[0]

    if fallback:
        return fallback[0]

    return None


def pick_defaults(chart_type: str, df: pd.DataFrame, a: dict) -> dict:
    """Suggests sensible default x/y/category columns per chart type,
    biased away from ID columns and toward meaningfully-named metrics."""
    cols = df.columns.tolist()

    if chart_type == "line":
        x = _first_available(a["date_cols"], fallback=cols)
        y = _first_available(a["money_cols"], a["count_cols"], a["numeric_non_id"], fallback=cols)
        return {"x": x, "y": y}

    if chart_type == "pie":
        low_card = [c for c in a["categorical_non_id"] if df[c].nunique(dropna=True) <= 12]
        names = _first_available(low_card, a["name_cols"], a["categorical_non_id"], fallback=cols)
        values = _first_available(a["money_cols"], a["count_cols"], a["numeric_non_id"], fallback=cols)
        return {"names": names, "values": values}

    if chart_type == "scatter":
        pool = a["numeric_non_id"] or df.select_dtypes(include="number").columns.tolist() or cols
        x = _first_available(a["money_cols"], fallback=pool)
        remaining = [c for c in pool if c != x]
        y = _first_available(a["count_cols"], remaining, fallback=pool)
        return {"x": x, "y": y}

    if chart_type == "histogram":
        x = _first_available(a["money_cols"], a["numeric_non_id"], fallback=cols)
        return {"x": x}

    # bar (default)
    x = _first_available(a["name_cols"], a["categorical_non_id"], fallback=cols)
    y = _first_available(a["money_cols"], a["count_cols"], a["numeric_non_id"], fallback=cols)
    return {"x": x, "y": y}


def build_hover_data(df: pd.DataFrame, a: dict, exclude: list) -> dict:
    """Builds a hover_data spec with sensible number formatting instead of
    Plotly's raw defaults — money columns get 2 decimals + thousands
    separator, counts get thousands separators, everything else shown as-is.
    ID columns are excluded from hover since they're not meaningful to a
    reader (still shown in the Data tab if needed)."""
    hover = {}
    for col in df.columns:
        if col in exclude:
            continue
        if col in a["id_cols"]:
            hover[col] = False  # hide raw ID clutter from tooltips
        elif col in a["money_cols"]:
            hover[col] = ":,.2f"
        elif col in a["count_cols"]:
            hover[col] = ":,"
        else:
            hover[col] = True
    return hover


def pretty_label(col: str) -> str:
    """Turns snake_case column names into Title Case for chart axis labels."""
    if col is None:
        return ""
    return col.replace("_", " ").title()


# ============================================================================
# BIG-SCREEN LOADER — one clean surface, human-readable only
# ============================================================================
# Deliberately does NOT expose raw SQL, tool arguments, or JSON results —
# that content lives in the SQL tab after the answer is ready, where the
# user can actually read it at their own pace, not flash past in a
# fast-moving technical log. What's shown here is a short, fixed list of
# CONCEPTUAL stages, each one lighting up as the matching tool actually
# runs — plausible progress the user can follow, not a raw trace.
_STAGE_LABELS = [
    ("understand", "🧠", "Understanding your question"),
    ("lookup", "📚", "Looking up business context"),
    ("schema", "📐", "Checking table structure"),
    ("resolve", "🔎", "Matching your wording to real data"),
    ("query", "⚙️", "Running your query"),
]

# Maps a real tool name to the conceptual stage it belongs to — several
# tools can map to the same stage (e.g. resolve_filter_value AND
# find_value_anywhere both mean "matching your wording to real data").
_TOOL_TO_STAGE = {
    "search_knowledge_base": "lookup",
    "list_tables": "schema",
    "describe_table": "schema",
    "resolve_filter_value": "resolve",
    "find_value_anywhere": "resolve",
    "get_sample_values": "resolve",
    "run_sql_query": "query",
}


def render_loader(placeholder, active_stage: str, done_stages: set):
    """Renders the ENTIRE loading experience as one HTML blob into
    `placeholder` — headline + fixed stage list, with each stage's CSS
    class reflecting done/active/pending. No st.status(), no separate
    container competing for the same visual space."""
    rows = []
    for key, icon, label in _STAGE_LABELS:
        if key in done_stages and key != active_stage:
            cls, shown_icon = "is-done", "✓"
        elif key == active_stage:
            cls, shown_icon = "is-active", icon
        else:
            cls, shown_icon = "", "○"
        rows.append(
            f'<div class="loading-step {cls}">'
            f'<span class="loading-step-icon">{shown_icon}</span>{label}</div>'
        )

    placeholder.markdown(
        f"""
        <div class="loading-overlay">
            <div class="loading-card">
                <div class="loading-spinner-big"></div>
                <div class="loading-headline">Working on your answer...</div>
                <div class="loading-steps">{''.join(rows)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_loader_step_handler(placeholder):
    """Returns an on_step(step) callback that advances the big-screen
    loader's stage list as real tool calls stream in. Multiple different
    tools can map to the same conceptual stage — that stage is marked
    active as soon as ANY of its tools starts, and marked done once we
    move on to a later stage (so a fast tool doesn't visibly flicker)."""
    state = {"active": "understand", "done": set()}

    def on_step(step: dict):
        if step["status"] != "started":
            return  # only stage TRANSITIONS matter for this display
        stage = _TOOL_TO_STAGE.get(step["tool_name"])
        if stage is None or stage == state["active"]:
            return
        state["done"].add(state["active"])
        state["active"] = stage
        render_loader(placeholder, state["active"], state["done"])

    return on_step


# ============================================================================
# LEARNING MODE — triggered only when the user's message starts with
# "error:". This is a distinct request type (teaching the agent a
# correction, not asking a normal question), so it gets its own detection,
# its own loader copy/color, and its own stage list built around
# save_learning/log_decision instead of the normal query-answering tools.
# ============================================================================
LEARNING_TRIGGER_PREFIX = "error:"


def is_learning_prompt(text: str) -> bool:
    """A prompt is a learning/correction request only if it EXPLICITLY
    starts with the trigger prefix (case-insensitive, allowing leading
    whitespace) — never inferred from wording alone, so an ordinary
    question that happens to mention the word "error" does not
    accidentally switch modes."""
    return text.strip().lower().startswith(LEARNING_TRIGGER_PREFIX)


_LEARNING_STAGE_LABELS = [
    ("diagnose", "🩺", "Diagnosing what went wrong"),
    ("lookup", "📚", "Checking schema and documented rules"),
    ("verify", "🔎", "Verifying the correct value/column"),
    ("save", "💾", "Saving this as a learned correction"),
]

# Maps a real tool name to the learning-mode conceptual stage it belongs
# to. save_learning/log_decision are the tools unique to this mode; the
# rest are the same schema/verification tools used in normal mode, just
# grouped differently here since the NARRATIVE is "diagnose and fix",
# not "answer a question".
_LEARNING_TOOL_TO_STAGE = {
    "search_learnings": "diagnose",
    "search_knowledge_base": "lookup",
    "list_tables": "lookup",
    "describe_table": "lookup",
    "run_sql_query": "verify",
    "save_learning": "save",
    "log_decision": "save",
}


def render_learning_loader(placeholder, active_stage: str, done_stages: set):
    """Same structure as render_loader, but amber-themed and with
    learning-specific copy, so it's visually unmistakable that the agent
    is in correction/learning mode rather than answering a new question."""
    rows = []
    for key, icon, label in _LEARNING_STAGE_LABELS:
        if key in done_stages and key != active_stage:
            cls, shown_icon = "is-done", "✓"
        elif key == active_stage:
            cls, shown_icon = "is-active", icon
        else:
            cls, shown_icon = "", "○"
        rows.append(
            f'<div class="loading-step learning-step {cls}">'
            f'<span class="loading-step-icon">{shown_icon}</span>{label}</div>'
        )

    placeholder.markdown(
        f"""
        <div class="loading-overlay">
            <div class="loading-card learning-card">
                <div class="learning-spinner-big"></div>
                <div>
                    <div class="learning-badge">🎓 Learning mode</div>
                    <div class="loading-headline">Teaching the agent this correction...</div>
                </div>
                <div class="loading-steps">{''.join(rows)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_learning_step_handler(placeholder):
    """Same pattern as make_loader_step_handler, but drives the
    learning-mode stage list and tool mapping instead."""
    state = {"active": "diagnose", "done": set()}

    def on_step(step: dict):
        if step["status"] != "started":
            return
        stage = _LEARNING_TOOL_TO_STAGE.get(step["tool_name"])
        if stage is None or stage == state["active"]:
            return
        state["done"].add(state["active"])
        state["active"] = stage
        render_learning_loader(placeholder, state["active"], state["done"])

    return on_step


# ============================================================================
# EXAMPLE PROMPTS (shown before any query has been run)
# ============================================================================
prompt = st.chat_input("Ask anything about your data... (start with 'error:' to correct a wrong answer)")

# ============================================================================
# RUN QUERY
# ============================================================================
if prompt:
    logger.info("User submitted prompt: %r", prompt)

    df = pd.DataFrame()
    qp = None
    query_error = None

    learning_mode = is_learning_prompt(prompt)

    loader_placeholder = st.empty()
    if learning_mode:
        logger.info("Learning-mode prompt detected (starts with %r)", LEARNING_TRIGGER_PREFIX)
        render_learning_loader(loader_placeholder, "diagnose", set())
        on_step = make_learning_step_handler(loader_placeholder)
    else:
        render_loader(loader_placeholder, "understand", set())
        on_step = make_loader_step_handler(loader_placeholder)

    try:
        # ----------------------------------------------------------------
        # STEP 1: PLAN
        # ----------------------------------------------------------------
        try:
            qp = plan(
                prompt,
                session_id=st.session_state.agent_session_id,
                on_step=on_step,
            )
            logger.info(
                "Plan generated successfully. refusal=%s sql=%s",
                qp.is_refusal, bool(qp.sql_used),
            )
        except Exception as e:
            query_error = e
            logger.exception("Planning failed for prompt: %r", prompt)
            qp = None

        # ----------------------------------------------------------------
        # STEP 2: EXECUTE SQL (if any)
        # ----------------------------------------------------------------
        if qp is not None and query_error is None and qp.sql_used and not qp.is_refusal:
            if learning_mode:
                render_learning_loader(loader_placeholder, "save", {"diagnose", "lookup", "verify"})
            else:
                render_loader(loader_placeholder, "query", {"understand", "lookup", "schema", "resolve"})
            try:
                df = run_sql(qp.sql_used)
                logger.info("SQL executed successfully: %d rows, %d columns", len(df), len(df.columns))
            except Exception as e:
                query_error = e
                logger.exception("SQL execution failed.\nSQL: %s", qp.sql_used)
                df = pd.DataFrame()

    except Exception as e:
        query_error = e
        logger.exception("Unexpected error while processing prompt: %r", prompt)

    finally:
        # Loader is the ONLY thing on screen during processing — always
        # clear it, success or failure, so the app never gets stuck showing
        # a spinner after the request has actually finished.
        try:
            loader_placeholder.empty()
        except Exception:
            logger.exception("Failed to remove loading overlay")

    # =========================================================================
    # HANDLE ERROR
    # =========================================================================
    if query_error is not None:
        # Keep the prompt visible even on failure — the user should still
        # see what they asked, not just a bare error with no context for
        # what triggered it.
        st.session_state.current_prompt = prompt
        st.session_state.current_is_learning = learning_mode
        st.chat_message("user").write(prompt)

        if learning_mode:
            st.error(f"❌ Learning-mode correction failed to process: {query_error}")
        else:
            st.error(f"❌ Something went wrong while processing your request: {query_error}")
        with st.expander("🔍 Full error details", expanded=True):
            st.exception(query_error)
        st.session_state.plan = None
        st.session_state.df = None

    # =========================================================================
    # SUCCESS / REFUSAL / EXPLANATION RESULT
    # =========================================================================
    elif qp is not None:
        st.session_state.plan = qp
        st.session_state.df = df
        st.session_state.current_prompt = prompt
        st.session_state.current_is_learning = learning_mode

        st.session_state.history.append(
            {
                "prompt": prompt,
                "chart_type": qp.chart_type if not qp.is_refusal else "refused",
                "rows": len(df),
                "query_plan": qp,
                "dataframe": df,
                "time": datetime.now().strftime("%H:%M"),
                "is_learning": learning_mode,
            }
        )

# ============================================================================
# RESULTS
# ============================================================================
if st.session_state.plan:
    qp = st.session_state.plan
    df = st.session_state.df

    if st.session_state.current_prompt:
        st.chat_message("user").write(st.session_state.current_prompt)

    if st.session_state.get("current_is_learning"):
        st.markdown(
            '<span class="learning-badge">🎓 Learning mode — correction taught to the agent</span>',
            unsafe_allow_html=True,
        )

    if qp.is_refusal:
        st.warning(f"🚫 {qp.explanation}")
    elif not qp.sql_used:
        st.info(f"ℹ️ {qp.explanation}")
    else:
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", len(df.columns))
        c.metric("Suggested chart", qp.chart_type.title() if not df.empty else "—")

        if show_insights and getattr(qp, "insights", None):
            st.subheader("💡 Insights")
            for i in qp.insights:
                st.markdown(f"<div class='card'>{i}</div>", unsafe_allow_html=True)

        if df.empty:
            st.info(
                "✅ Query ran successfully — **0 rows matched** your question. "
                "This is a real result, not an error. Check the SQL tab below to "
                "see exactly what was searched for."
            )
            with st.expander("📝 SQL", expanded=True):
                st.code(qp.sql_used, language="sql")
                st.caption(qp.explanation)
        elif len(df.columns) == 0:
            st.error("No columns were returned.")
        else:
            analysis = analyze_columns(df)
            t1, t2, t3 = st.tabs(["📈 Visualization", "📄 Data", "📝 SQL"])

            with t1:
                cols = list(df.columns)
                nums = df.select_dtypes(include="number").columns.tolist()
                cats = [c for c in cols if c not in nums]

                chart_types = ["bar", "line", "pie", "scatter", "histogram"]
                default_chart = qp.chart_type if qp.chart_type in chart_types else "bar"
                ct = st.selectbox("Chart Type", chart_types, index=chart_types.index(default_chart))

                defaults = pick_defaults(ct, df, analysis)
                for key in defaults:
                    if defaults[key] is None:
                        defaults[key] = df.columns[0]

                col1, col2, col3 = st.columns(3)

                if ct == "bar":
                    x = col1.selectbox("X Axis", cats or cols, index=(cats or cols).index(defaults["x"]) if defaults["x"] in (cats or cols) else 0)
                    y = col2.selectbox("Y Axis", nums or cols, index=(nums or cols).index(defaults["y"]) if defaults["y"] in (nums or cols) else 0)
                    color_options = ["None"] + [c for c in cols if c not in [x, y]]
                    color = col3.selectbox("Color By", color_options)
                    hover = build_hover_data(df, analysis, exclude=[x, y])
                    labels = {}
                    if x is not None:
                        labels[x] = pretty_label(x)
                    if y is not None:
                        labels[y] = pretty_label(y)

                    fig = px.bar(
                        df, x=x, y=y,
                        color=None if color == "None" else color,
                        hover_data=hover, barmode="group",
                        color_discrete_sequence=PALETTE, labels=labels,
                    )
                elif ct == "line":
                    x = col1.selectbox("X Axis", cols, index=cols.index(defaults["x"]) if defaults["x"] in cols else 0)
                    y = col2.selectbox("Y Axis", nums or cols, index=(nums or cols).index(defaults["y"]) if defaults["y"] in (nums or cols) else 0)
                    suggested_color = next((c for c in cols if c != x and c in analysis["name_cols"]), "None")
                    color_options = ["None"] + [c for c in cols if c != x]
                    color = col3.selectbox("Color By", color_options,
                                            index=color_options.index(suggested_color) if suggested_color in color_options else 0)
                    hover = build_hover_data(df, analysis, exclude=[x, y])
                    labels = {}
                    if x is not None:
                        labels[x] = pretty_label(x)
                    if y is not None:
                        labels[y] = pretty_label(y)

                    fig = px.line(
                        df, x=x, y=y,
                        color=None if color == "None" else color,
                        markers=True, hover_data=hover,
                        color_discrete_sequence=PALETTE, labels=labels,
                    )
                elif ct == "pie":
                    x = col1.selectbox("Category", cats or cols, index=(cats or cols).index(defaults["names"]) if defaults["names"] in (cats or cols) else 0)
                    y = col2.selectbox("Value", nums or cols, index=(nums or cols).index(defaults["values"]) if defaults["values"] in (nums or cols) else 0)
                    hover = build_hover_data(df, analysis, exclude=[x, y])
                    fig = px.pie(df, names=x, values=y, hover_data=hover, color_discrete_sequence=PALETTE)
                    fig.update_traces(textinfo="percent+label")
                elif ct == "scatter":
                    x = col1.selectbox("X Axis", nums or cols, index=(nums or cols).index(defaults["x"]) if defaults["x"] in (nums or cols) else 0)
                    y_options = nums or cols
                    y_default_idx = y_options.index(defaults["y"]) if defaults["y"] in y_options else min(1, len(y_options) - 1)
                    y = col2.selectbox("Y Axis", y_options, index=y_default_idx)
                    color_options = ["None"] + [c for c in cols if c not in [x, y]]
                    color = col3.selectbox("Color By", color_options)
                    hover = build_hover_data(df, analysis, exclude=[x, y])
                    labels = {}
                    if x is not None:
                        labels[x] = pretty_label(x)
                    if y is not None:
                        labels[y] = pretty_label(y)

                    fig = px.scatter(
                        df, x=x, y=y,
                        color=None if color == "None" else color,
                        hover_data=hover, color_discrete_sequence=PALETTE, labels=labels,
                    )
                else:  # histogram
                    x = col1.selectbox("Column", nums or cols, index=(nums or cols).index(defaults["x"]) if defaults["x"] in (nums or cols) else 0)
                    color_options = ["None"] + [c for c in cols if c != x]
                    color = col3.selectbox("Color By", color_options)
                    hover = build_hover_data(df, analysis, exclude=[x])
                    labels = {}
                    if x is not None:
                        labels[x] = pretty_label(x)

                    fig = px.histogram(
                        df, x=x,
                        color=None if color == "None" else color,
                        hover_data=hover, color_discrete_sequence=PALETTE, labels=labels,
                    )

                fig.update_layout(
                    template="plotly_white", height=560, hovermode="closest",
                    legend_title_text="", font=dict(family="Inter, sans-serif", size=13),
                    margin=dict(t=30, l=10, r=10, b=10),
                )
            st.plotly_chart(fig, width='stretch')

            with t2:
                st.dataframe(df, width="stretch")
                st.download_button("📥 Download CSV", df.to_csv(index=False), "results.csv", "text/csv")

            with t3:
                if show_sql:
                    st.code(qp.sql_used, language="sql")
                    st.caption(qp.explanation)