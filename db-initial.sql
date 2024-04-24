--------------------
-- SCHEMA CORE
--------------------
-- Name: core; Type: SCHEMA; Schema: -; Owner: postgres
DROP SCHEMA IF EXISTS "core" CASCADE;

CREATE SCHEMA IF NOT EXISTS "core";

CREATE USER airflow_user WITH PASSWORD 'NnWMfNsOXCXsvv8lWQhQaelyCRfGZQGH7bUgjyebZNv1PANPbxoXbZPhvTCtm8HV';
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow_user;
-- PostgreSQL 15 requires additional privileges:
USE airflow_db;
GRANT ALL ON SCHEMA core TO airflow_user;

ALTER USER airflow_user SET search_path = core;