-- ==============================================================================
-- Aegis Procure: Oracle AI Database 26ai Schema Initialization
-- Tables: VENDORS, PURCHASE_ORDERS, CONTRACT_CHUNKS
-- Vector Index: HNSW Cosine on CONTRACT_CHUNKS(embedding)
-- ==============================================================================

-- Drop tables if they already exist (in reverse foreign-key order)
BEGIN
    EXECUTE IMMEDIATE 'DROP TABLE CONTRACT_CHUNKS CASCADE CONSTRAINTS';
EXCEPTION
    WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; END IF;
END;
/

BEGIN
    EXECUTE IMMEDIATE 'DROP TABLE PURCHASE_ORDERS CASCADE CONSTRAINTS';
EXCEPTION
    WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; END IF;
END;
/

BEGIN
    EXECUTE IMMEDIATE 'DROP TABLE VENDORS CASCADE CONSTRAINTS';
EXCEPTION
    WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; END IF;
END;
/

-- 1. Structured ERP Table: VENDORS
CREATE TABLE VENDORS (
    vendor_id       VARCHAR2(50) PRIMARY KEY,
    name            VARCHAR2(255) NOT NULL,
    tier            VARCHAR2(20) CHECK (tier IN ('Tier-1', 'Tier-2', 'Tier-3')),
    risk_rating     VARCHAR2(20) CHECK (risk_rating IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    status          VARCHAR2(30) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'UNDER_REVIEW', 'SUSPENDED')),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Structured ERP Table: PURCHASE_ORDERS
CREATE TABLE PURCHASE_ORDERS (
    po_id            VARCHAR2(50) PRIMARY KEY,
    vendor_id        VARCHAR2(50) NOT NULL REFERENCES VENDORS(vendor_id),
    amount           NUMBER(14, 2) NOT NULL,
    status           VARCHAR2(30) CHECK (status IN ('PENDING', 'APPROVED', 'PAID', 'DISPUTED', 'CANCELLED')),
    order_date       DATE NOT NULL,
    item_description VARCHAR2(1000) NOT NULL,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Converged Vector & Relational Table: CONTRACT_CHUNKS
-- Native 384-dimensional vector column matching all-MiniLM-L6-v2
CREATE TABLE CONTRACT_CHUNKS (
    chunk_id         VARCHAR2(50) PRIMARY KEY,
    vendor_id        VARCHAR2(50) NOT NULL REFERENCES VENDORS(vendor_id),
    clause_type      VARCHAR2(50) NOT NULL, -- e.g. SLA, LIABILITY, PAYMENT_TERMS, TERMINATION, CONFIDENTIALITY
    clause_text      CLOB NOT NULL,
    embedding        VECTOR(384, FLOAT32),
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. In-Database Vector Search Index: HNSW (Cosine Distance)
CREATE VECTOR INDEX contract_chunks_hnsw_idx 
ON CONTRACT_CHUNKS(embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- Helpful index for hybrid queries filtering by vendor_id and clause_type
CREATE INDEX idx_contract_chunks_vendor ON CONTRACT_CHUNKS(vendor_id, clause_type);
CREATE INDEX idx_po_vendor_date ON PURCHASE_ORDERS(vendor_id, order_date);
