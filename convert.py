from sqlalchemy import (
    create_engine,
    inspect,
    MetaData,
    Table,
    Column,
    Integer,
    Float,
    String,
    Text,
    Date,
    DateTime,
    Numeric,
    ForeignKey,
)
import pandas as pd

###########################################################
# Connections
###########################################################

mssql_engine = create_engine(
    r"mssql+pyodbc://analytics_app_login:pass123@MuhammadAdnan\SQLEXPRESS/BikeStores?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
)

sqlite_engine = create_engine("sqlite:///BikeStores2.db")

inspector = inspect(mssql_engine)

metadata = MetaData()

###########################################################
# SQL Server -> SQLite datatype mapping
###########################################################

def map_type(sqltype):

    t = str(sqltype).upper()

    if "INT" in t:
        return Integer

    if "DECIMAL" in t or "NUMERIC" in t:
        return Numeric

    if "FLOAT" in t or "REAL" in t:
        return Float

    if "DATE" in t and "TIME" not in t:
        return Date

    if "DATETIME" in t or "TIME" in t:
        return DateTime

    if "CHAR" in t or "TEXT" in t or "VARCHAR" in t or "NVARCHAR" in t:
        return Text

    return Text


###########################################################
# Create tables
###########################################################

tables = inspector.get_table_names()

for table_name in tables:

    print(f"Creating schema for {table_name}")

    columns = inspector.get_columns(table_name)

    pk = inspector.get_pk_constraint(table_name)

    fk = inspector.get_foreign_keys(table_name)

    fk_lookup = {}

    for f in fk:

        for local, remote in zip(
            f["constrained_columns"],
            f["referred_columns"],
        ):

            fk_lookup[local] = (
                f"{f['referred_table']}.{remote}"
            )

    table_columns = []

    pk_columns = pk.get("constrained_columns", [])

    composite_pk = len(pk_columns) > 1

    for c in columns:

        args = []

        if c["name"] in fk_lookup:
            args.append(ForeignKey(fk_lookup[c["name"]]))

        col = Column(
            c["name"],
            map_type(c["type"]),
            *args,
            primary_key=(not composite_pk and c["name"] in pk_columns),
            nullable=c["nullable"],
        )

        table_columns.append(col)

    if composite_pk:

        table = Table(
            table_name,
            metadata,
            *table_columns,
        )

        table.primary_key.columns.clear()

        for col in table_columns:
            if col.name in pk_columns:
                col.primary_key = True

    else:

        Table(
            table_name,
            metadata,
            *table_columns,
        )

###########################################################
# Create SQLite schema
###########################################################

metadata.create_all(sqlite_engine)

###########################################################
# Copy data
###########################################################

for table_name in tables:

    print(f"Copying {table_name}")

    df = pd.read_sql(
        f"SELECT * FROM [{table_name}]",
        mssql_engine,
    )

    df.to_sql(
        table_name,
        sqlite_engine,
        if_exists="append",
        index=False,
    )

print("\nFinished!")