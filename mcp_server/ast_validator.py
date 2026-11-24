"""
Aegis Procure: SQL AST Guard & Query Validator
Enforces strictly read-only SELECT queries on whitelisted ERP tables.
Uses SQLGlot for robust AST analysis and injection defense.
"""

from typing import Tuple, Set

try:
    import sqlglot
    from sqlglot import exp
except ImportError:
    sqlglot = None
    exp = None


# Whitelist of allowed enterprise procurement tables
ALLOWED_TABLES: Set[str] = {
    "VENDORS",
    "PURCHASE_ORDERS",
    "CONTRACT_CHUNKS",
    "DUAL"
}

# Forbidden keywords / DDL / DML patterns
FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", 
    "TRUNCATE", "REPLACE", "GRANT", "REVOKE", "MERGE", "CALL",
    "EXECUTE", "BEGIN", "DECLARE", "SYS", "DBA_", "V$", "GV$"
}


def validate_sql_query(query: str) -> Tuple[bool, str, str]:
    """
    Validates a SQL query using AST parsing.
    Returns:
        (is_valid: bool, error_message: str, parsed_sql: str)
    """
    if not query or not query.strip():
        return False, "Query is empty.", ""

    clean_query = query.strip().rstrip(";")

    # Fast keyword check
    upper_query = clean_query.upper()
    for kw in ["DROP ", "DELETE ", "UPDATE ", "INSERT ", "TRUNCATE ", "ALTER ", "EXEC "]:
        if kw in upper_query:
            return False, f"Security Violation: Mutating keyword '{kw.strip()}' is strictly prohibited.", ""

    # If sqlglot is available, perform strict AST analysis
    if sqlglot is not None:
        try:
            parsed_statements = sqlglot.parse(clean_query, read="oracle")
        except Exception as e:
            return False, f"SQL Parse Error: Malformed query syntax ({str(e)})", ""

        # Enforce single-statement execution
        if len(parsed_statements) != 1:
            return False, "Security Violation: Multiple chained SQL statements detected.", ""

        statement = parsed_statements[0]
        if statement is None:
            return False, "Security Violation: Empty or unparseable statement.", ""

        # Rule: Must be a pure SELECT statement
        if not isinstance(statement, exp.Select):
            return False, f"Security Violation: Only SELECT queries are permitted. Detected: {type(statement).__name__}", ""

        # Rule: Validate accessed tables against whitelist
        tables = set()
        for table in statement.find_all(exp.Table):
            table_name = table.name.upper()
            tables.add(table_name)
            if table_name not in ALLOWED_TABLES:
                return False, f"Access Denied: Table '{table_name}' is not in the authorized ERP whitelist.", ""

        # Rule: Prevent access to Oracle metadata dictionaries
        for col in statement.find_all(exp.Column):
            col_upper = col.name.upper()
            if col_upper.startswith("V$") or col_upper.startswith("DBA_") or col_upper.startswith("ALL_"):
                return False, f"Security Violation: Access to system catalog '{col_upper}' is blocked.", ""

        return True, "Query authorized.", statement.sql(dialect="oracle")

    # Fallback when sqlglot is not installed (basic heuristic guard)
    tokens = clean_query.split()
    if not tokens or tokens[0].upper() != "SELECT":
        return False, "Security Violation: Only SELECT queries are permitted.", ""

    for kw in FORBIDDEN_KEYWORDS:
        if kw in upper_query:
            return False, f"Security Violation: Prohibited keyword '{kw}' detected.", ""

    return True, "Query authorized (heuristic mode).", clean_query


if __name__ == "__main__":
    # Self-test cases
    test_queries = [
        ("SELECT * FROM VENDORS WHERE tier = 'Tier-1'", True),
        ("SELECT vendor_id, SUM(amount) FROM PURCHASE_ORDERS GROUP BY vendor_id", True),
        ("DROP TABLE VENDORS", False),
        ("SELECT * FROM VENDORS; DELETE FROM PURCHASE_ORDERS", False),
        ("SELECT * FROM ALL_USERS", False),
        ("UPDATE VENDORS SET status = 'ACTIVE'", False)
    ]

    for q, expected in test_queries:
        valid, msg, _ = validate_sql_query(q)
        status = "PASS" if valid == expected else "FAIL"
        print(f"[{status}] Query: {q} -> Valid: {valid} ({msg})")
