"""
Aegis Procure: Data Ingestion & Contract Vectorization Pipeline
Converged Oracle 26ai Free Ingestion:
  1. Relational ERP data (VENDORS, PURCHASE_ORDERS) -> SQL Bulk Insert
  2. Unstructured Contract Clauses -> Local CPU embedding (all-MiniLM-L6-v2)
     -> array('f', ...) -> Oracle VECTOR(384, FLOAT32) column.
"""

import os
import sys
import csv
import json
import argparse
from array import array
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Third-party dependencies
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import oracledb
except ImportError:
    oracledb = None

# Default local connection parameters for Oracle AI Database 26ai Free (Docker)
DB_CONFIG = {
    "user": os.getenv("ORACLE_USER", "system"),
    "password": os.getenv("ORACLE_PASSWORD", "AegisProcure2024!"),
    "host": os.getenv("ORACLE_HOST", "localhost"),
    "port": int(os.getenv("ORACLE_PORT", "1521")),
    "service_name": os.getenv("ORACLE_SERVICE", "FREEPDB1"),
}

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384


def get_db_connection():
    """Establishes a Thin Mode connection to Oracle AI Database 26ai."""
    if oracledb is None:
        raise ImportError("python-oracledb is not installed. Please run: pip install oracledb")
    
    # python-oracledb uses Thin Mode by default (no Oracle Client libraries needed)
    dsn = f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['service_name']}"
    print(f"[*] Connecting to Oracle 26ai (Thin Mode): {DB_CONFIG['user']}@{dsn}")
    connection = oracledb.connect(
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        dsn=dsn
    )
    return connection


def load_embedding_model() -> Any:
    """Loads all-MiniLM-L6-v2 locally on CPU."""
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed. Please run: pip install sentence-transformers")
    print(f"[*] Loading local embedding model: '{EMBEDDING_MODEL_NAME}' on CPU...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return model


def embed_texts(model: Any, texts: List[str]) -> List[array]:
    """
    Generates 384-dimensional normalized vector embeddings on CPU
    and converts them to array('f', ...) for direct python-oracledb binding.
    """
    print(f"[*] Computing vector embeddings for {len(texts)} contract clauses...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    
    # Convert numpy float arrays into Python native array('f')
    vector_arrays = [array("f", emb.tolist()) for emb in embeddings]
    return vector_arrays


def load_from_json(json_path: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Loads vendors, purchase orders, and contracts from unified JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    vendors = []
    purchase_orders = []
    contracts = []

    for item in data:
        if "vendor" in item:
            vendors.append(item["vendor"])
        if "purchase_orders" in item:
            purchase_orders.extend(item["purchase_orders"])
        if "contracts" in item:
            contracts.extend(item["contracts"])

    return vendors, purchase_orders, contracts


def load_from_csv(ingest_dir: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Loads vendors, purchase orders, and contract clauses from separate CSV files."""
    vendors = []
    purchase_orders = []
    contracts = []

    v_file = ingest_dir / "sample_vendors.csv"
    po_file = ingest_dir / "sample_purchase_orders.csv"
    c_file = ingest_dir / "sample_contract_clauses.csv"

    if v_file.exists():
        with open(v_file, "r", encoding="utf-8") as f:
            vendors = list(csv.DictReader(f))

    if po_file.exists():
        with open(po_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["amount"] = float(row["amount"])
                purchase_orders.append(row)

    if c_file.exists():
        with open(c_file, "r", encoding="utf-8") as f:
            contracts = list(csv.DictReader(f))

    return vendors, purchase_orders, contracts


def ingest_data(vendors: List[Dict], purchase_orders: List[Dict], contracts: List[Dict], dry_run: bool = False):
    """
    Embeds contract clauses and inserts all records into Oracle 26ai tables.
    """
    print(f"\n[+] Processing Dataset:")
    print(f"    - Vendors: {len(vendors)}")
    print(f"    - Purchase Orders: {len(purchase_orders)}")
    print(f"    - Contract Clauses: {len(contracts)}")

    # 1. Compute embeddings for contract clauses
    if contracts:
        model = load_embedding_model()
        clause_texts = [c["clause_text"] for c in contracts]
        embeddings = embed_texts(model, clause_texts)
        
        # Attach array('f') vector to each contract chunk
        for c, emb in zip(contracts, embeddings):
            c["embedding"] = emb

    if dry_run:
        print("\n[✔] DRY RUN COMPLETE: Data parsed and embeddings generated successfully.")
        print(f"    Sample vector length: {len(contracts[0]['embedding'])} floats")
        print(f"    Sample vector format: {type(contracts[0]['embedding'])} ({contracts[0]['embedding'].typecode})")
        return

    # 2. Database Insertion
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # A. Upsert VENDORS
        print("[*] Inserting VENDORS...")
        vendor_sql = """
            MERGE INTO VENDORS v
            USING (SELECT :1 AS vendor_id, :2 AS name, :3 AS tier, :4 AS risk_rating, :5 AS status FROM dual) src
            ON (v.vendor_id = src.vendor_id)
            WHEN MATCHED THEN
                UPDATE SET v.name = src.name, v.tier = src.tier, v.risk_rating = src.risk_rating, v.status = src.status
            WHEN NOT MATCHED THEN
                INSERT (vendor_id, name, tier, risk_rating, status)
                VALUES (src.vendor_id, src.name, src.tier, src.risk_rating, src.status)
        """
        vendor_params = [
            (v["vendor_id"], v["name"], v["tier"], v["risk_rating"], v["status"])
            for v in vendors
        ]
        cursor.executemany(vendor_sql, vendor_params)
        print(f"[✔] Ingested {len(vendor_params)} vendor records.")

        # B. Upsert PURCHASE_ORDERS
        print("[*] Inserting PURCHASE_ORDERS...")
        po_sql = """
            MERGE INTO PURCHASE_ORDERS p
            USING (SELECT :1 AS po_id, :2 AS vendor_id, :3 AS amount, :4 AS status, 
                          TO_DATE(:5, 'YYYY-MM-DD') AS order_date, :6 AS item_description FROM dual) src
            ON (p.po_id = src.po_id)
            WHEN MATCHED THEN
                UPDATE SET p.amount = src.amount, p.status = src.status, 
                           p.order_date = src.order_date, p.item_description = src.item_description
            WHEN NOT MATCHED THEN
                INSERT (po_id, vendor_id, amount, status, order_date, item_description)
                VALUES (src.po_id, src.vendor_id, src.amount, src.status, src.order_date, src.item_description)
        """
        po_params = [
            (po["po_id"], po["vendor_id"], float(po["amount"]), po["status"], str(po["order_date"]), po["item_description"])
            for po in purchase_orders
        ]
        cursor.executemany(po_sql, po_params)
        print(f"[✔] Ingested {len(po_params)} purchase order records.")

        # C. Upsert CONTRACT_CHUNKS with native VECTOR(384, FLOAT32)
        print("[*] Inserting CONTRACT_CHUNKS with native 384-dim vectors...")
        contract_sql = """
            MERGE INTO CONTRACT_CHUNKS c
            USING (SELECT :1 AS chunk_id, :2 AS vendor_id, :3 AS clause_type, :4 AS clause_text, :5 AS embedding FROM dual) src
            ON (c.chunk_id = src.chunk_id)
            WHEN MATCHED THEN
                UPDATE SET c.clause_type = src.clause_type, c.clause_text = src.clause_text, c.embedding = src.embedding
            WHEN NOT MATCHED THEN
                INSERT (chunk_id, vendor_id, clause_type, clause_text, embedding)
                VALUES (src.chunk_id, src.vendor_id, src.clause_type, src.clause_text, src.embedding)
        """
        contract_params = [
            (c["chunk_id"], c["vendor_id"], c["clause_type"], c["clause_text"], c["embedding"])
            for c in contracts
        ]
        cursor.executemany(contract_sql, contract_params)
        print(f"[✔] Ingested {len(contract_params)} contract chunks with HNSW-ready vector embeddings.")

        conn.commit()
        print("\n[✔] ALL INGESTION SUCCESSFULLY COMMITTED TO ORACLE 26ai.")

    except Exception as e:
        conn.rollback()
        print(f"\n[✘] Error during database ingestion: {e}", file=sys.stderr)
        raise
    finally:
        cursor.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Aegis Procure: Ingest contracts and ERP data into Oracle 26ai.")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="Data source format (default: json)")
    parser.add_argument("--dry-run", action="store_true", help="Generate embeddings and validate data without connecting to Oracle DB")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if args.format == "json":
        json_file = base_dir / "sample_contracts.json"
        if not json_file.exists():
            print(f"[!] Error: {json_file} does not exist.")
            sys.exit(1)
        vendors, purchase_orders, contracts = load_from_json(json_file)
    else:
        vendors, purchase_orders, contracts = load_from_csv(base_dir)

    ingest_data(vendors, purchase_orders, contracts, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
