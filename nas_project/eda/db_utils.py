import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus

def get_connection_string(credentials):
    """Constructs the SQLAlchemy connection string for PostgreSQL."""
    return f"postgresql+psycopg2://{credentials['user']}:{quote_plus(credentials['password'])}@{credentials['host']}:{credentials['port']}/{credentials['database']}"

def test_connection(credentials):
    """Tests the database connection."""
    try:
        engine = create_engine(get_connection_string(credentials))
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True, None
    except Exception as e:
        return False, str(e)

def get_schemas(credentials):
    """Retrieves a list of schemas from the database."""
    try:
        engine = create_engine(get_connection_string(credentials))
        query = "SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')"
        with engine.connect() as connection:
            result = connection.execute(text(query))
            schemas = [row[0] for row in result]
        return schemas, None
    except Exception as e:
        return None, str(e)

def get_tables(credentials, schema):
    """Retrieves a list of tables in a specific schema."""
    try:
        engine = create_engine(get_connection_string(credentials))
        query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}'"
        with engine.connect() as connection:
            result = connection.execute(text(query))
            tables = [row[0] for row in result]
        return tables, None
    except Exception as e:
        return None, str(e)

def execute_query(credentials, query, preview=True):
    """Executes a SQL query and returns a DataFrame."""
    try:
        engine = create_engine(get_connection_string(credentials))
        
        if preview:
            # Wrap query to limit results for preview
            # Note: This is a simple limit, might need more robust parsing for complex queries
            # but for now we'll just append LIMIT if it's a SELECT
            if "limit" not in query.lower() and query.strip().lower().startswith("select"):
                query += " LIMIT 100"
        
        df = pd.read_sql(query, engine)
        return df, None
    except Exception as e:
        return None, str(e)
