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

/* Centered full-screen loading overlay */
.loading-overlay {{
    position: fixed;
    inset: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(2px);
}}
.loading-box {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 14px;
    padding: 28px 36px;
    border-radius: 16px;
    background: white;
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
}}
.loading-spinner {{
    width: 42px;
    height: 42px;
    border: 4px solid #eef2ff;
    border-top: 4px solid {ACCENT};
    border-radius: 50%;
    animation: loading-spin 0.8s linear infinite;
}}
.loading-text {{
    font-weight: 600;
    color: #374151;
    font-size: 0.95rem;
}}
@keyframes loading-spin {{
    0% {{ transform: rotate(0deg); }}
    100% {{ transform: rotate(360deg); }}
}}
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
        st.rerun()
    if b2.button("🗑 Reset", width='stretch', help="Clear everything including history"):
        st.session_state.history = []
        st.session_state.plan = None
        st.session_state.df = None
        st.session_state.current_prompt = None
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
        # Prefer a low-cardinality categorical column for slices.
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
# EXAMPLE PROMPTS (shown before any query has been run)
# ============================================================================
# EXAMPLES = [
#     "Total revenue by store",
#     "Top 10 best-selling products",
#     "Monthly order trend this year",
#     "Which customers spent the most?",
# ]

prompt = st.chat_input("Ask anything about your data...")
# if not prompt:
#     prompt = st.session_state.pop("example", None)
# if not prompt:
#     prompt = st.session_state.pop("force_prompt", None)

# if st.session_state.plan is None and not prompt:
#     st.markdown("##### Try one of these:")
#     st.markdown('<div class="chip-row">', unsafe_allow_html=True)
#     cols = st.columns(len(EXAMPLES))
#     for c, ex in zip(cols, EXAMPLES):
#         if c.button(ex, width='stretch'):
#             st.session_state["example"] = ex
#             st.rerun()
#     st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# RUN QUERY
# ============================================================================
if prompt:
    logger.info("User submitted prompt: %r", prompt)

    # ------------------------------------------------------------------------
    # Initialize state
    # ------------------------------------------------------------------------
    df = pd.DataFrame()
    qp = None
    query_error = None

    # ------------------------------------------------------------------------
    # Create loading overlay
    # ------------------------------------------------------------------------
    loader_placeholder = st.empty()

    loader_placeholder.markdown(
        """
        <div class="loading-overlay">
            <div class="loading-box">
                <div class="loading-spinner"></div>
                <div class="loading-text">
                    Analyzing your question...
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        # ====================================================================
        # AI PROCESSING
        # ====================================================================
        with st.status("🤖 AI is thinking...", expanded=True) as status:

            # ----------------------------------------------------------------
            # STEP 1: PLAN
            # ----------------------------------------------------------------
            try:
                st.write("🔍 Understanding your question...")

                qp = plan(prompt)

                logger.info(
                    "Plan generated successfully. refusal=%s sql=%s",
                    qp.is_refusal,
                    bool(qp.sql_used),
                )

            except Exception as e:
                query_error = e

                logger.exception(
                    "Planning failed for prompt: %r",
                    prompt,
                )

                status.update(
                    label="❌ Failed while generating the query plan",
                    state="error",
                )

                # Stop processing this request, but DON'T raise.
                qp = None

            # ----------------------------------------------------------------
            # STEP 2: PROCESS PLAN
            # ----------------------------------------------------------------
            if qp is not None and query_error is None:

                # ============================================================
                # REFUSAL
                # ============================================================
                if qp.is_refusal:
                    logger.info(
                        "Request refused: %s",
                        qp.explanation,
                    )

                    status.update(
                        label="Request cannot be fulfilled",
                        state="error",
                    )

                    df = pd.DataFrame()

                # ============================================================
                # NO SQL
                # ============================================================
                elif not qp.sql_used:
                    logger.info(
                        "No SQL generated. Explanation: %s",
                        qp.explanation,
                    )

                    status.update(
                        label="Completed",
                        state="complete",
                    )

                    df = pd.DataFrame()

                # ============================================================
                # SQL GENERATED
                # ============================================================
                else:

                    st.write("⚙️ Executing SQL...")

                    try:
                        # ----------------------------------------------------
                        # Execute SQL
                        # ----------------------------------------------------
                        df = run_sql(qp.sql_used)

                        logger.info(
                            "SQL executed successfully: %d rows, %d columns",
                            len(df),
                            len(df.columns),
                        )

                        status.update(
                            label="✅ Query completed",
                            state="complete",
                        )

                    except Exception as e:
                        query_error = e

                        logger.exception(
                            "SQL execution failed.\nSQL: %s",
                            qp.sql_used,
                        )

                        status.update(
                            label="❌ SQL execution failed",
                            state="error",
                        )

                        # IMPORTANT:
                        # Do NOT `raise` here.
                        #
                        # Raising the exception would terminate execution
                        # before the loader cleanup code.
                        df = pd.DataFrame()

    except Exception as e:
        # ====================================================================
        # UNEXPECTED ERROR
        # ====================================================================
        query_error = e

        logger.exception(
            "Unexpected error while processing prompt: %r",
            prompt,
        )

    finally:
        # ====================================================================
        # CRITICAL LOADER CLEANUP
        #
        # This ALWAYS executes:
        #   - success
        #   - SQL error
        #   - LLM error
        #   - retrieval error
        #   - unexpected exception
        # ====================================================================
        try:
            loader_placeholder.empty()
        except Exception:
            logger.exception("Failed to remove loading overlay")

    # =========================================================================
    # HANDLE ERROR AFTER LOADER HAS BEEN REMOVED
    # =========================================================================
    if query_error is not None:

        st.error(
            "❌ Something went wrong while processing your request."
        )

        # ------------------------------------------------------------
        # Show technical error details in an expandable section.
        # This is useful during development but doesn't clutter the UI.
        # ------------------------------------------------------------
        with st.expander("🔍 Error details", expanded=False):
            st.exception(query_error)

        # ------------------------------------------------------------
        # Clear failed query state
        # ------------------------------------------------------------
        st.session_state.plan = None
        st.session_state.df = None
        st.session_state.current_prompt = None

    # =========================================================================
    # SUCCESS / REFUSAL / EXPLANATION RESULT
    # =========================================================================
    elif qp is not None:

        # ---------------------------------------------------------------------
        # Save current result
        # ---------------------------------------------------------------------
        st.session_state.plan = qp
        st.session_state.df = df
        st.session_state.current_prompt = prompt

        # ---------------------------------------------------------------------
        # Add to history
        # ---------------------------------------------------------------------
        st.session_state.history.append(
            {
                "prompt": prompt,
                "chart_type": (
                    qp.chart_type
                    if not qp.is_refusal
                    else "refused"
                ),
                "rows": len(df),
                "query_plan": qp,
                "dataframe": df,
                "time": datetime.now().strftime("%H:%M"),
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

    if qp.is_refusal:
        st.warning(f"🚫 {qp.explanation}")
    elif not qp.sql_used:
        # No SQL was actually executed — the agent answered from its lookup
        # tools directly instead of writing a real query. This is NOT the
        # same as "query ran and matched 0 rows" and must not be shown as
        # such — show the explanation plainly instead.
        st.info(f"ℹ️ {qp.explanation}")
    else:
        # --- CHANGED: previously called st.stop() the instant df was empty,
        # which hid EVERYTHING below it — including the SQL tab. That made a
        # perfectly legitimate "zero matches" answer look like something had
        # broken, since there was no way to see what SQL actually ran or
        # confirm it was a real (not buggy) empty result. Now we still show
        # metrics/insights/SQL, and only skip the parts that don't make
        # sense for zero rows (the chart + data table). ---
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
    # Ensure defaults always exist
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
                        df,
                        x=x,
                        y=y,
                        color=None if color == "None" else color,
                        hover_data=hover,
                        barmode="group",
                        color_discrete_sequence=PALETTE,
                        labels=labels,
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
                        df,
                        x=x,
                        y=y,
                        color=None if color == "None" else color,
                        markers=True,
                        hover_data=hover,
                        color_discrete_sequence=PALETTE,
                        labels=labels,
                    )

                elif ct == "pie":
                    x = col1.selectbox("Category", cats or cols, index=(cats or cols).index(defaults["names"]) if defaults["names"] in (cats or cols) else 0)
                    y = col2.selectbox("Value", nums or cols, index=(nums or cols).index(defaults["values"]) if defaults["values"] in (nums or cols) else 0)
                    hover = build_hover_data(df, analysis, exclude=[x, y])
                    fig = px.pie(
                        df, names=x, values=y, hover_data=hover,
                        color_discrete_sequence=PALETTE,
                    )
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
                        df,
                        x=x,
                        y=y,
                        color=None if color == "None" else color,
                        hover_data=hover,
                        color_discrete_sequence=PALETTE,
                        labels=labels,
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
                        df,
                        x=x,
                        color=None if color == "None" else color,
                        hover_data=hover,
                        color_discrete_sequence=PALETTE,
                        labels=labels,
                    )

                fig.update_layout(
                    template="plotly_white",
                    height=560,
                    hovermode="closest",
                    legend_title_text="",
                    font=dict(family="Inter, sans-serif", size=13),
                    margin=dict(t=30, l=10, r=10, b=10),
                )
                st.caption(f"Debug: figure has {len(fig.data)} trace(s)")  # TEMPORARY — remove once confirmed working
            st.plotly_chart(fig, width='stretch')

            with t2:
                st.dataframe(df, width="stretch")
                st.download_button("📥 Download CSV", df.to_csv(index=False), "results.csv", "text/csv")

            with t3:
                if show_sql:
                    st.code(qp.sql_used, language="sql")
                    st.caption(qp.explanation)