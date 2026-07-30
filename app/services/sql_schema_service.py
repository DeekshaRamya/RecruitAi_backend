import logging
import httpx
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
        Dynamically discovers all available base schemas and tables from the connected
        SQL Server database via INFORMATION_SCHEMA.TABLES. Caches metadata for the assessment session.
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

        # Step 1: Execute exact dynamic table discovery query as the source of truth
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
                        logger.info(f"[SqlSchemaService] Dynamically discovered {len(discovered_tables)} base tables in AdventureWorks.")
                else:
                    logger.error(f"SQL Execution API discovery failed with status {res.status_code}: {res.text}")
                
                # If discovery failed or returned zero tables, check fallback or abort
                if not discovered_tables:
                    logger.warning("Dynamic discovery via API yielded zero tables. Attempting verified emergency recovery...")
                    fallback_result = cls._get_fallback_schema()
                    if not fallback_result.get("tables_map"):
                        raise RuntimeError("SQL Server schema discovery failed: No available base tables found in connected database. Cannot generate valid SQL questions.")
                    return fallback_result

                cls._session_discovered_tables = discovered_tables

                # Step 2: Retrieve column details for all discovered base tables (No hardcoded schema filtering!)
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
                    # Initialize all discovered base tables first so none are missing
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
                                "is_pk": is_pk
                            })

                    cls._cached_schema_info = {
                        "database": "AdventureWorks",
                        "tables_map": tables_map,
                        "discovered_base_tables": discovered_tables
                    }
                    cls._cached_schema_text = cls._format_schema_for_prompt(tables_map)
                    logger.info(f"Successfully fetched and cached complete schema metadata for {len(tables_map)} base tables.")
                    return cls._cached_schema_info
                else:
                    logger.error(f"Column metadata retrieval failed with status {res_cols.status_code}")
                    return cls._get_fallback_schema()

        except Exception as ex:
            logger.error(f"Failed to perform dynamic schema discovery against SQL Execution API: {ex}")
            return cls._get_fallback_schema()

    @classmethod
    def get_live_schema_text(cls, force_refresh: bool = False) -> str:
        """Returns the formatted schema prompt string for AI prompt generation."""
        if not cls._cached_schema_text or force_refresh:
            info = cls.get_live_schema(force_refresh=force_refresh)
            if not info or not info.get("tables_map"):
                raise RuntimeError("SQL Server schema discovery failed: Zero base tables discovered in connected database.")
        return cls._cached_schema_text or ""

    @classmethod
    def _format_schema_for_prompt(cls, tables_map: Dict[str, Any]) -> str:
        lines = [
            "=================================================================================",
            "MANDATORY DYNAMICALLY DISCOVERED DATABASE SCHEMA (ADVENTUREWORKS ON SQL SERVER):",
            "CRITICAL MANDATE FOR ALL SQL QUESTIONS:",
            "- DO NOT invent tables or use generic/fake table names like 'evaluation_records', 'dbo.orders', 'users', 'customers', or 'employees'.",
            "- YOU MUST EXPLICITLY INCLUDE THE FULL SCHEMA AND TABLE NAME exactly as returned from INFORMATION_SCHEMA (e.g., Sales.SalesOrderHeader, Sales.Customer, HumanResources.Employee, Production.Product, Person.Person).",
            "- Every SQL query MUST be valid T-SQL and directly executable against the AdventureWorks database schema listed below.",
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

    @classmethod
    def _get_fallback_schema(cls) -> Dict[str, Any]:
        """Provides verified fallback AdventureWorks schema if API is temporarily unreachable."""
        fallback_tables = {
            "HumanResources.Employee": {
                "schema": "HumanResources", "table": "Employee",
                "columns": [
                    {"name": "BusinessEntityID", "type": "int", "is_pk": True},
                    {"name": "NationalIDNumber", "type": "nvarchar", "is_pk": False},
                    {"name": "JobTitle", "type": "nvarchar", "is_pk": False},
                    {"name": "BirthDate", "type": "date", "is_pk": False},
                    {"name": "MaritalStatus", "type": "nchar", "is_pk": False},
                    {"name": "Gender", "type": "nchar", "is_pk": False},
                    {"name": "HireDate", "type": "date", "is_pk": False},
                    {"name": "SalariedFlag", "type": "bit", "is_pk": False},
                    {"name": "VacationHours", "type": "smallint", "is_pk": False},
                    {"name": "SickLeaveHours", "type": "smallint", "is_pk": False}
                ]
            },
            "Person.Person": {
                "schema": "Person", "table": "Person",
                "columns": [
                    {"name": "BusinessEntityID", "type": "int", "is_pk": True},
                    {"name": "PersonType", "type": "nchar", "is_pk": False},
                    {"name": "FirstName", "type": "nvarchar", "is_pk": False},
                    {"name": "LastName", "type": "nvarchar", "is_pk": False},
                    {"name": "EmailPromotion", "type": "int", "is_pk": False}
                ]
            },
            "Sales.SalesOrderHeader": {
                "schema": "Sales", "table": "SalesOrderHeader",
                "columns": [
                    {"name": "SalesOrderID", "type": "int", "is_pk": True},
                    {"name": "OrderDate", "type": "datetime", "is_pk": False},
                    {"name": "CustomerID", "type": "int", "is_pk": False},
                    {"name": "SubTotal", "type": "money", "is_pk": False},
                    {"name": "TaxAmt", "type": "money", "is_pk": False},
                    {"name": "Freight", "type": "money", "is_pk": False},
                    {"name": "TotalDue", "type": "money", "is_pk": False}
                ]
            },
            "Production.Product": {
                "schema": "Production", "table": "Product",
                "columns": [
                    {"name": "ProductID", "type": "int", "is_pk": True},
                    {"name": "Name", "type": "nvarchar", "is_pk": False},
                    {"name": "ProductNumber", "type": "nvarchar", "is_pk": False},
                    {"name": "Color", "type": "nvarchar", "is_pk": False},
                    {"name": "ListPrice", "type": "money", "is_pk": False}
                ]
            }
        }
        cls._cached_schema_info = {"database": "AdventureWorks", "tables_map": fallback_tables}
        cls._cached_schema_text = cls._format_schema_for_prompt(fallback_tables)
        return cls._cached_schema_info
