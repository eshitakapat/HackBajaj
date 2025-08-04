--
-- PostgreSQL database dump
--

-- Dumped from database version 14.18 (Ubuntu 14.18-0ubuntu0.22.04.1)
-- Dumped by pg_dump version 14.18 (Ubuntu 14.18-0ubuntu0.22.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


--
-- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: clauses; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.clauses (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    document_id uuid,
    clause_text text,
    clause_index integer,
    embedding public.vector(1536),
    created_at timestamp without time zone DEFAULT now()
);


ALTER TABLE public.clauses OWNER TO postgres;

--
-- Name: documents; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.documents (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    title text,
    source_type text,
    domain text,
    upload_time timestamp without time zone DEFAULT now(),
    original_file_path text,
    processed boolean DEFAULT false,
    CONSTRAINT documents_domain_check CHECK ((domain = ANY (ARRAY['insurance'::text, 'legal'::text, 'hr'::text, 'compliance'::text]))),
    CONSTRAINT documents_source_type_check CHECK ((source_type = ANY (ARRAY['pdf'::text, 'docx'::text, 'email'::text])))
);


ALTER TABLE public.documents OWNER TO postgres;

--
-- Name: clauses clauses_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.clauses
    ADD CONSTRAINT clauses_pkey PRIMARY KEY (id);


--
-- Name: documents documents_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_pkey PRIMARY KEY (id);


--
-- Name: idx_clause_doc_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_clause_doc_id ON public.clauses USING btree (document_id);


--
-- Name: idx_clause_embedding; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_clause_embedding ON public.clauses USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='100');


--
-- Name: clauses clauses_document_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.clauses
    ADD CONSTRAINT clauses_document_id_fkey FOREIGN KEY (document_id) REFERENCES public.documents(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

