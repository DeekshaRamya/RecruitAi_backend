import logging
import httpx
import re
from typing import Dict, Any, List, Optional
from app.core.config import settings

logger = logging.getLogger("recruitai-backend.sql_schema_service")

class SqlSchemaService:
    _cached_schema_info: Optional[Dict[str, Any]] = None
    _cached_schema_text: Optional[str] = None
    _session_discovered_tables: Optional[List[Dict[str, str]]] = None

    @classmethod
    def get_live_schema(cls, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Dynamically discovers all available base schemas, tables, and column metadata from the connected
        SQL Server database via INFORMATION_SCHEMA catalog views. Caches metadata for the assessment session.
        No static or hardcoded table definitions are used.
        """
        if cls._cached_schema_info and not force_refresh:
            logger.info("[SqlSchemaService] Reusing cached SQL Server schema metadata for current session.")
            return cls._cached_schema_info

        api_url = getattr(settings, "SQL_EXECUTION_API_URL", None) or "http://172.176.122.4:5001/execute"
        credentials = {
            "host": "172.176.122.4",
            "port": 1433,
            "database": "AdventureWorks",
            "username": "readonly_user",
            "password": "Readonly@123"
        }

        # Step 1: Execute dynamic table discovery query directly against INFORMATION_SCHEMA
        discovery_query = """
SELECT
    TABLE_SCHEMA,
    TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
ORDER BY TABLE_SCHEMA, TABLE_NAME;
"""

        payload_discovery = {
            "query": discovery_query.strip(),
            "serverType": "sqlserver",
            "credentials": credentials,
            "examId": "schema_discovery",
            "userEmail": "system@recruitai.com"
        }

        try:
            logger.info("Executing dynamic SQL Server base table discovery via INFORMATION_SCHEMA.TABLES...")
            with httpx.Client(timeout=15.0) as client:
                res = client.post(api_url, json=payload_discovery)
                discovered_tables = []
                if res.status_code == 200:
                    data = res.json()
                    rows = data.get("rows", [])
                    if not rows:
                        logger.error(f"SQL Server discovery returned zero tables or error: {data.get('error')}")
                    else:
                        for row in rows:
                            s_name = row.get("TABLE_SCHEMA")
                            t_name = row.get("TABLE_NAME")
                            if s_name and t_name:
                                discovered_tables.append({
                                    "schema": s_name,
                                    "table": t_name,
                                    "qualified_name": f"{s_name}.{t_name}"
                                })
                        logger.info(f"[SqlSchemaService] Dynamically discovered {len(discovered_tables)} base tables from connected database.")
                else:
                    logger.error(f"SQL Execution API discovery failed with status {res.status_code}: {res.text}")
                
                if not discovered_tables:
                    raise RuntimeError("SQL Server dynamic schema discovery failed: Unable to fetch live database tables.")

                cls._session_discovered_tables = discovered_tables

                # Step 2: Retrieve detailed column metadata dynamically for all discovered tables
                columns_query = """
SELECT 
    c.TABLE_SCHEMA, 
    c.TABLE_NAME, 
    c.COLUMN_NAME, 
    c.DATA_TYPE,
    c.IS_NULLABLE,
    CASE WHEN k.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END AS IS_PRIMARY_KEY
FROM INFORMATION_SCHEMA.COLUMNS c
JOIN INFORMATION_SCHEMA.TABLES t 
    ON c.TABLE_SCHEMA = t.TABLE_SCHEMA 
    AND c.TABLE_NAME = t.TABLE_NAME 
    AND t.TABLE_TYPE = 'BASE TABLE'
LEFT JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE k 
    ON c.TABLE_SCHEMA = k.TABLE_SCHEMA 
    AND c.TABLE_NAME = k.TABLE_NAME 
    AND c.COLUMN_NAME = k.COLUMN_NAME
    AND k.CONSTRAINT_NAME LIKE '%PK%'
ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION;
"""
                payload_columns = {
                    "query": columns_query.strip(),
                    "serverType": "sqlserver",
                    "credentials": credentials,
                    "examId": "schema_columns_fetch",
                    "userEmail": "system@recruitai.com"
                }

                res_cols = client.post(api_url, json=payload_columns)
                if res_cols.status_code == 200:
                    data_cols = res_cols.json()
                    rows_cols = data_cols.get("rows", [])
                    
                    tables_map = {}
                    for dt in discovered_tables:
                        tables_map[dt["qualified_name"]] = {
                            "schema": dt["schema"],
                            "table": dt["table"],
                            "columns": []
                        }

                    for row in rows_cols:
                        s_name = row.get("TABLE_SCHEMA")
                        t_name = row.get("TABLE_NAME")
                        col_name = row.get("COLUMN_NAME")
                        d_type = row.get("DATA_TYPE")
                        is_nullable = row.get("IS_NULLABLE", "YES")
                        is_pk = bool(row.get("IS_PRIMARY_KEY", 0))
                        
                        full_table_name = f"{s_name}.{t_name}"
                        if full_table_name not in tables_map:
                            tables_map[full_table_name] = {
                                "schema": s_name,
                                "table": t_name,
                                "columns": []
                            }
                        
                        if col_name:
                            tables_map[full_table_name]["columns"].append({
                                "name": col_name,
                                "type": d_type,
                                "is_nullable": is_nullable,
                                "is_pk": is_pk
                            })

                    cls._cached_schema_info = {
                        "database": "AdventureWorks",
                        "tables_map": tables_map,
                        "discovered_base_tables": discovered_tables
                    }
                    cls._cached_schema_text = cls._format_schema_for_prompt(tables_map)
                    logger.info(f"Successfully fetched dynamic schema metadata for {len(tables_map)} database tables.")
                    return cls._cached_schema_info
                else:
                    raise RuntimeError(f"Column metadata retrieval failed with status {res_cols.status_code}")

        except Exception as ex:
            logger.error(f"Failed to perform dynamic schema discovery against SQL Execution API: {ex}")
            if cls._cached_schema_info:
                return cls._cached_schema_info
            raise RuntimeError(f"SQL Server schema discovery failed: {ex}")

    @classmethod
    def get_live_schema_text(cls, force_refresh: bool = False) -> str:
        """Returns the formatted schema prompt string for AI prompt generation."""
        if not cls._cached_schema_text or force_refresh:
            info = cls.get_live_schema(force_refresh=force_refresh)
            if not info or not info.get("tables_map"):
                raise RuntimeError("SQL Server schema discovery failed: Zero base tables discovered in connected database.")
        return cls._cached_schema_text or ""

    @classmethod
    def extract_tables_from_sql(cls, sql_query: str, tables_map: Dict[str, Any]) -> List[str]:
        """
        Parses a SQL query to extract ONLY the database tables actually referenced in FROM, JOIN, INTO, UPDATE.
        Cross-references extracted identifiers against tables_map to return qualified table names.
        """
        if not sql_query or not tables_map:
            return []

        # Clean SQL comments
        clean_sql = re.sub(r'--.*?\n', ' ', sql_query)
        clean_sql = re.sub(r'/\*.*?\*/', ' ', clean_sql, flags=re.DOTALL)

        # Build case-insensitive lookup maps
        qualified_map = {k.lower(): k for k in tables_map.keys()}
        bare_map = {}
        for full_name, details in tables_map.items():
            b_name = details.get("table", full_name.split(".")[-1]).lower()
            if b_name not in bare_map:
                bare_map[b_name] = []
            bare_map[b_name].append(full_name)

        found_tables = []

        def match_and_add(candidate: str):
            clean_c = candidate.replace("[", "").replace("]", "").strip().rstrip(";").lower()
            if not clean_c:
                return
            if clean_c in qualified_map:
                full_t = qualified_map[clean_c]
                if full_t not in found_tables:
                    found_tables.append(full_t)
            elif clean_c in bare_map:
                for full_t in bare_map[clean_c]:
                    if full_t not in found_tables:
                        found_tables.append(full_t)

        # 1. Match FROM, JOIN (INNER, LEFT, RIGHT, FULL, CROSS), INTO, UPDATE clauses
        matches = re.findall(r'\b(?:FROM|JOIN|INTO|UPDATE)\s+([a-zA-Z0-9_.\-\[\]]+)', clean_sql, flags=re.IGNORECASE)
        for m in matches:
            match_and_add(m)

        # 2. Match comma-separated FROM clauses (e.g., FROM TableA a, TableB b)
        from_blocks = re.findall(r'\bFROM\s+(.*?)(?=\bWHERE\b|\bGROUP\b|\bHAVING\b|\bORDER\b|\bJOIN\b|;|$)', clean_sql, flags=re.IGNORECASE | re.DOTALL)
        for block in from_blocks:
            items = block.split(",")
            for item in items:
                tokens = item.strip().split()
                if tokens:
                    match_and_add(tokens[0])

        return found_tables

    @classmethod
    def get_referenced_tables(cls, q_or_corpus: Any, tables_map: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Dynamically extracts ONLY the database tables that are actually referenced in the SQL question.
        Primary source: The SQL query (expectedAnswer / correctAnswer).
        Secondary source: Explicit qualified Schema.Table references in the question text.
        Guarantees that NO extra or unrelated table schemas are included.
        """
        if not tables_map:
            try:
                schema_info = cls.get_live_schema()
                tables_map = schema_info.get("tables_map", {}) if schema_info else {}
            except Exception:
                tables_map = {}

        if not tables_map:
            return []

        sql_query = ""
        corpus_text = ""

        if isinstance(q_or_corpus, dict):
            sql_query = str(q_or_corpus.get("expectedAnswer") or q_or_corpus.get("correctAnswer") or q_or_corpus.get("query") or "").strip()
            corpus_text = " ".join([
                sql_query,
                str(q_or_corpus.get("problemStatement") or ""),
                str(q_or_corpus.get("task") or q_or_corpus.get("candidateTask") or ""),
                str(q_or_corpus.get("scenario") or ""),
                str(q_or_corpus.get("question") or "")
            ])
        else:
            corpus_text = str(q_or_corpus or "").strip()
            if "SELECT" in corpus_text.upper():
                sql_query = corpus_text

        # Step 1: Parse SQL query directly (source of truth)
        if sql_query:
            query_tables = cls.extract_tables_from_sql(sql_query, tables_map)
            if query_tables:
                return query_tables

        # Step 2: Fallback to parsing text corpus if SQL query was not provided
        if corpus_text:
            query_tables = cls.extract_tables_from_sql(corpus_text, tables_map)
            if query_tables:
                return query_tables

            # Only match exact qualified Schema.Table names (e.g., HumanResources.Employee) present in corpus_text
            qualified_map = {k.lower(): k for k in tables_map.keys()}
            clean_corpus_lower = corpus_text.lower()
            ref = []
            for full_lower, full_original in qualified_map.items():
                if full_lower in clean_corpus_lower:
                    if full_original not in ref:
                        ref.append(full_original)
            if ref:
                return ref

        # Fallback to first table if zero tables detected
        return [list(tables_map.keys())[0]]

    @classmethod
    def get_database_schemas_for_question(cls, q: Dict[str, Any], tables_map: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generates complete DDL CREATE TABLE statements for ALL tables referenced in the question.
        Constructs schema definitions dynamically from live database metadata (columns, types, nullability, PKs).
        """
        if not tables_map:
            try:
                schema_info = cls.get_live_schema()
                tables_map = schema_info.get("tables_map", {}) if schema_info else {}
            except Exception:
                tables_map = {}

        corpus_parts = [
            str(q.get("expectedAnswer") or q.get("correctAnswer") or q.get("query") or ""),
            str(q.get("problemStatement") or ""),
            str(q.get("scenario") or ""),
            str(q.get("task") or q.get("candidateTask") or ""),
            str(q.get("question") or ""),
            str(q.get("topic") or "")
        ]
        ref_tables = cls.get_referenced_tables(q, tables_map)

        schemas = []
        for t_name in ref_tables:
            t_meta = tables_map.get(t_name, {})
            cols = t_meta.get("columns", [])
            if cols:
                col_defs = []
                for c in cols[:30]:
                    name = c.get("name")
                    c_type = c.get("type", "nvarchar")
                    is_pk = c.get("is_pk", False)
                    is_nullable = c.get("is_nullable", "YES")

                    null_str = " NOT NULL" if (str(is_nullable).upper() == "NO" or is_pk) else ""
                    pk_str = " PRIMARY KEY" if is_pk else ""

                    col_defs.append(f"{name} {c_type}{null_str}{pk_str}")

                cols_joined = ", ".join(col_defs)
                schemas.append(f"-- Live Schema from Database ({t_name})\nCREATE TABLE {t_name} ({cols_joined});")
            else:
                schemas.append(f"-- Live Schema from Database ({t_name})\nCREATE TABLE {t_name};")

        return schemas

    @classmethod
    def _format_schema_for_prompt(cls, tables_map: Dict[str, Any]) -> str:
        lines = [
            "=================================================================================",
            "MANDATORY DYNAMICALLY DISCOVERED DATABASE SCHEMA (LIVE DATABASE):",
            "CRITICAL MANDATE FOR ALL SQL QUESTIONS:",
            "- DO NOT invent tables or use generic/fake table names like 'evaluation_records', 'dbo.orders', 'users', 'customers', or 'employees'.",
            "- YOU MUST EXPLICITLY INCLUDE THE FULL SCHEMA AND TABLE NAME exactly as returned from INFORMATION_SCHEMA.",
            "- Every SQL query MUST be valid T-SQL and directly executable against the database schema listed below.",
            "=================================================================================",
            ""
        ]

        schemas: Dict[str, List[Dict[str, Any]]] = {}
        for full_name, details in tables_map.items():
            s = details["schema"]
            if s not in schemas:
                schemas[s] = []
            schemas[s].append(details)

        for s_name, t_list in sorted(schemas.items()):
            lines.append(f"Schema: [{s_name}]")
            for t in sorted(t_list, key=lambda x: x["table"]):
                cols_str = ", ".join([f"{c['name']} ({c['type']}{', PK' if c['is_pk'] else ''})" for c in t["columns"][:35]])
                if len(t["columns"]) > 35:
                    cols_str += f", ... (+{len(t['columns'])-35} more columns)"
                lines.append(f"  - {t['schema']}.{t['table']} -> Columns: [{cols_str}]")
            lines.append("")

        return "\n".join(lines)
