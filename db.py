import logging
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


DB_URL = os.environ.get("MYSQL_DB_URL")
# DB_URL = os.environ.get("MS_DB_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ENGINE = create_engine(DB_URL)
logger.info("engine created")
# ENGINE = create_engine("sqlite:///BikeStores2.db")
 
READ_ONLY_PREFIXES = ("select", "with")

FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "truncate",
    "merge", "exec", "execute", "grant", "revoke", "create",
)

def run_sql(sql: str):
    cleaned = sql.strip().rstrip(";")
    lowered = cleaned.lower()

    if not lowered.startswith(READ_ONLY_PREFIXES):
        logger.warning("Rejected non-SELECT/WITH SQL: %r", cleaned[:200])
        raise ValueError(
            f"Only SELECT/WITH (CTE) statements are allowed. Got: {cleaned[:80]!r}"
        )
    import re
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lowered):
            logger.warning("Rejected SQL containing forbidden keyword %r: %r", keyword, cleaned[:200])
            raise ValueError(f"Query contains a forbidden keyword: {keyword!r}")

    logger.info("Executing SQL: %s", cleaned)
    logger.debug("Executing SQL — repr: %r", cleaned)
    try:
        with ENGINE.connect().execution_options(no_parameters=True) as conn:
            df = pd.read_sql_query(cleaned, conn)
    except Exception:
        logger.exception("SQL execution failed for query: %s", cleaned)
        raise
    logger.info("SQL executed successfully — %d row(s), %d column(s)", len(df), len(df.columns))
    return df