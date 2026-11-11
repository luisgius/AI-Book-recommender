-- Schema of the cloud search profile (Postgres / Neon), version 1.
--
-- Versioned with the code and applied by app/infrastructure/db/postgres_schema.py
-- (python -m scripts.load_postgres). Every statement is idempotent.
--
-- Placeholders, replaced before execution:
--   __SCHEMA__   catalog schema (validated identifier; one schema per catalog,
--                so several catalogs can live in one database)
--   __DIM__      embedding dimension of the configured embedding model
--   __BM25_K1__  BM25 k1 (term-frequency saturation)
--   __BM25_B__   BM25 b (document-length normalisation)
--
-- Sections are separated by "-- @section <name>" lines and run one at a
-- time: the schema first, the indexes only after the bulk load (BM25 corpus
-- statistics are computed when its index is built).

-- @section extensions
-- Extensions are per database (not per schema) and shared by every catalog.
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS vector;
DO $$
BEGIN
  -- Lakebase Search (Neon, Postgres 16+): a real BM25 index (lakebase_text)
  -- and an ANN index (lakebase_vector). Optional on other Postgres servers.
  IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'lakebase_text') THEN
    CREATE EXTENSION IF NOT EXISTS lakebase_text;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'lakebase_vector') THEN
    CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE;
  END IF;
END $$;

-- @section schema
CREATE SCHEMA IF NOT EXISTS __SCHEMA__;

-- Facts about the loaded catalog (embedding model, index types, text-search
-- configurations in use, build times), written by the loader and read once
-- by the search adapters.
CREATE TABLE IF NOT EXISTS __SCHEMA__.catalog_info (
  key   text PRIMARY KEY,
  value jsonb NOT NULL
);

-- One row per work (a Book of the domain).
CREATE TABLE IF NOT EXISTS __SCHEMA__.works (
  id             uuid PRIMARY KEY,
  source         text NOT NULL,
  source_id      text,
  title          text NOT NULL,
  authors        text[] NOT NULL,
  description    text,                     -- primary description (Book.description)
  language       text,                     -- ISO 639-1 code of the work
  categories     text[] NOT NULL DEFAULT '{}',
  published_date timestamp,
  year           integer GENERATED ALWAYS AS (extract(year FROM published_date)::integer) STORED,
  metadata       jsonb,                    -- identifiers, descriptions, provenance... (BookMetadata)
  created_at     timestamptz NOT NULL,
  updated_at     timestamptz NOT NULL,
  content_hash   text NOT NULL,            -- change detection for idempotent reloads
  text_hash      text NOT NULL,            -- hash of the text the embedding was computed from
  embedding      halfvec(__DIM__),         -- unit-length document embedding (16-bit floats)
  UNIQUE (source, source_id)
);

CREATE INDEX IF NOT EXISTS works_language_idx ON __SCHEMA__.works (lower(language));
CREATE INDEX IF NOT EXISTS works_year_idx ON __SCHEMA__.works (year);

-- One row per description language of a work; a work without description
-- has one row with language ''. Each row is a complete lexical document of
-- the work: title, authors and categories (simple_unaccent, not stemmed)
-- plus the description (stemmed with the configuration of its language).
CREATE TABLE IF NOT EXISTS __SCHEMA__.work_texts (
  work_id   uuid NOT NULL REFERENCES __SCHEMA__.works (id) ON DELETE CASCADE,
  language  text NOT NULL,                 -- ISO 639-1 code of body, '' if unknown or none
  ts_config text NOT NULL,                 -- schema-qualified text-search configuration of body
  body      text,                          -- description in that language
  tsv       tsvector,                      -- filled by the loader (see postgres_schema.py)
  PRIMARY KEY (work_id, language)
);

-- Text-search configurations: one per stemmed language that this Postgres
-- ships (read from pg_catalog, not assumed), with unaccent in front of the
-- stemmer, plus simple_unaccent for titles, authors, categories and for
-- languages without a stemmer.
DO $$
DECLARE
  target    text := '__SCHEMA__';
  unaccent  text;
  cfg       record;
  chain     text;
BEGIN
  SELECT quote_ident(n.nspname) || '.unaccent' INTO unaccent
  FROM pg_extension e JOIN pg_namespace n ON n.oid = e.extnamespace
  WHERE e.extname = 'unaccent';

  FOR cfg IN
    SELECT c.oid, c.cfgname FROM pg_ts_config c
    WHERE c.cfgnamespace = 'pg_catalog'::regnamespace
    ORDER BY c.cfgname
  LOOP
    CONTINUE WHEN EXISTS (
      SELECT 1 FROM pg_ts_config x
      WHERE x.cfgnamespace = target::regnamespace AND x.cfgname = cfg.cfgname || '_unaccent'
    );
    -- Dictionary chain the shipped configuration uses for plain words.
    SELECT string_agg(m.mapdict::regdictionary::text, ', ' ORDER BY m.mapseqno) INTO chain
    FROM pg_ts_config_map m
    WHERE m.mapcfg = cfg.oid
      AND m.maptokentype = (SELECT tokid FROM ts_token_type('default') WHERE alias = 'asciiword');
    EXECUTE format('CREATE TEXT SEARCH CONFIGURATION %I.%I (COPY = pg_catalog.%I)',
                   target, cfg.cfgname || '_unaccent', cfg.cfgname);
    -- Words with non-ASCII letters: fold accents first, then the same chain.
    EXECUTE format('ALTER TEXT SEARCH CONFIGURATION %I.%I ALTER MAPPING FOR word, hword, hword_part WITH %s, %s',
                   target, cfg.cfgname || '_unaccent', unaccent, chain);
  END LOOP;
END $$;

-- Query side of the lexical search: the lexemes of a query under every
-- configuration in use. A query word that is a stopword in any of those
-- languages is dropped (the local tokenizer drops English and Spanish
-- stopwords the same way); the remaining words contribute their lexemes
-- under every configuration (unstemmed from simple_unaccent, stemmed from
-- each language), so they match titles and descriptions alike.
CREATE OR REPLACE FUNCTION __SCHEMA__.query_lexemes(query text, configs regconfig[])
RETURNS text[]
LANGUAGE sql STABLE PARALLEL SAFE
AS $fn$
  WITH tokens AS (
    SELECT d.token, d.lexemes
    FROM unnest(configs) AS c(cfg)
    CROSS JOIN LATERAL ts_debug(c.cfg, query) AS d
    WHERE d.lexemes IS NOT NULL
  ),
  stopwords AS (
    SELECT DISTINCT token FROM tokens WHERE cardinality(lexemes) = 0
  )
  SELECT coalesce(array_agg(DISTINCT lexeme ORDER BY lexeme), '{}'::text[])
  FROM tokens CROSS JOIN LATERAL unnest(tokens.lexemes) AS lexeme
  WHERE tokens.token NOT IN (SELECT token FROM stopwords)
$fn$;

-- OR of the given lexemes as a tsquery (any lexeme matches), for ts_rank_cd.
CREATE OR REPLACE FUNCTION __SCHEMA__.or_tsquery(lexemes text[])
RETURNS tsquery
LANGUAGE sql IMMUTABLE PARALLEL SAFE
AS $fn$
  SELECT coalesce(
    string_agg('''' || replace(replace(l, '\', '\\'), '''', '''''') || '''', ' | '),
    ''
  )::tsquery
  FROM unnest(lexemes) AS l
$fn$;

-- @section lexical_index_gin
-- Inverted index for the ts_rank_cd ranking (matching with @@).
CREATE INDEX IF NOT EXISTS work_texts_tsv_gin ON __SCHEMA__.work_texts USING gin (tsv);

-- @section lexical_index_bm25
-- BM25 index (lakebase_text). k1 and b are fixed when the index is built.
CREATE INDEX IF NOT EXISTS work_texts_bm25 ON __SCHEMA__.work_texts
  USING lakebase_bm25 (tsv) WITH (k1 = __BM25_K1__, b = __BM25_B__);

-- @section vector_index_lakebase_ann
-- Neon's ANN index (IVF with full-precision re-ranking; exact flat search
-- while the table is small). Cosine distance on halfvec.
CREATE INDEX IF NOT EXISTS works_embedding_ann ON __SCHEMA__.works
  USING lakebase_ann (embedding halfvec_cosine_ops);

-- @section vector_index_hnsw
-- pgvector's HNSW graph (pgvector defaults m = 16, ef_construction = 64).
CREATE INDEX IF NOT EXISTS works_embedding_ann ON __SCHEMA__.works
  USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);
