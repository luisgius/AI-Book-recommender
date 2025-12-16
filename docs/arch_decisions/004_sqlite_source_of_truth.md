# ADR 004: SQLite as the Source of Truth for the Book Catalog

## Status

Accepted

## Context

Our hybrid book recommendation system needs a persistence layer for storing book data imported from external APIs (Google Books, Open Library). This persistence layer must:

1. Serve as the **canonical source** for all book information
2. Support efficient CRUD operations for book entities
3. Enable rebuilding of search indices (BM25, FAISS) from stored data
4. Handle deduplication when re-importing books from external sources
5. Integrate cleanly with our hexagonal architecture

The key architectural question is: **Where does authoritative book data live?**

Options considered:
- Store books only in search indices (BM25/FAISS)
- Store books in both SQLite AND indices, with indices as primary
- Store books in SQLite as source of truth, with indices as derived artifacts

## Decision

We will use **SQLite as the single source of truth** for all book data, implemented via `SqliteBookCatalogRepository`.

### Key Design Choices

#### 1. SQLite is Authoritative, Indices are Derived

```
    SQLite (books table)           BM25 Index          FAISS Index
    ====================           ==========          ===========
    [Source of Truth]      --->    [Derived]    --->   [Derived]

    - All book data                - Built from        - Built from
    - Survives restarts            SQLite              SQLite
    - Canonical identity           - Rebuildable       - Rebuildable
                                   - In-memory         - Persistable
```

If the BM25 or FAISS index becomes corrupted or out-of-sync, we simply rebuild from SQLite:

```python
books = catalog_repo.get_all()
bm25_repo.build_index(books)
embeddings_store.build_index()
```

#### 2. Denormalized Schema (Single Table)

We use a single `books` table with JSON-encoded arrays instead of normalized tables:

```sql
CREATE TABLE books (
    id INTEGER PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,        -- JSON array: ["Author 1", "Author 2"]
    categories TEXT,              -- JSON array: ["Fiction", "Classic"]
    metadata_json TEXT,           -- JSON object: BookMetadata fields
    ...
    UNIQUE(source, source_id)
);
```

**Why denormalized?**
- Simpler queries (no JOINs needed)
- Faster reads (single table scan)
- Easier to understand for thesis documentation
- Acceptable trade-off for read-heavy workload with moderate catalog size

**Trade-offs accepted:**
- Cannot query individual authors efficiently (e.g., "find all books by Author X")
- Data duplication if same author writes multiple books
- Harder to enforce author/category consistency

For this academic project with <10k books and primarily search-based access patterns, denormalization is the right choice.

#### 3. Deduplication Strategy: (source, source_id)

External APIs assign their own IDs to books:
- Google Books: `"abc123xyz"`
- Open Library: `"OL12345M"`

We use the composite key `(source, source_id)` as a **natural key** for deduplication:

```
UNIQUE(source, source_id)
```

When importing a book:
1. Check if `(source, source_id)` exists
2. If YES: **UPDATE** existing row (upsert)
3. If NO: **INSERT** new row

This prevents:
- Duplicate entries when re-running ingestion
- Conflicts between IDs from different sources (same source_id, different sources = different books)

#### 4. UUID Preservation on Upsert

**Critical invariant:** The domain UUID must never change for an existing book.

```python
# In _upsert_book():
if existing is not None:
    # UPDATE: preserve existing uuid, update other fields
    existing.title = book.title
    existing.authors = ...
    # NOTE: uuid is NOT updated
else:
    # INSERT: use the book's uuid
    row = BookRow(uuid=str(book.id), ...)
```

**Why preserve UUID?**

The BM25 and FAISS indices reference books by UUID:
- BM25: `book_id -> Book` mapping
- FAISS: `index_position -> UUID -> Book`

If we replaced the UUID during re-import:
1. Indices would point to non-existent UUIDs
2. Search results would fail to hydrate
3. The system would be inconsistent

By preserving UUID, indices remain valid even after catalog updates.

#### 5. Data Type Mappings

| Domain Type | SQLite Column | Notes |
|-------------|---------------|-------|
| `UUID` | `TEXT` | SQLite lacks native UUID; stored as string |
| `List[str]` | `TEXT` (JSON) | `["a", "b"]` with `ensure_ascii=False` |
| `datetime` | `TEXT` (ISO8601) | `2024-01-15T10:30:00+00:00` |
| `BookMetadata` | `TEXT` (JSON) | Nullable, sparse storage |

## Rationale

### Why SQLite (not PostgreSQL, MySQL, etc.)?

1. **Zero operational overhead**: No server to install or manage
2. **Single file**: Easy to backup, share, inspect
3. **Sufficient performance**: Handles 10k+ books easily
4. **Built into Python**: sqlite3 module is standard library
5. **Thesis-friendly**: Easy to explain, no infrastructure dependencies

### Why Source of Truth Architecture?

1. **Resilience**: Indices can be rebuilt from SQLite at any time
2. **Consistency**: Single authoritative source prevents data drift
3. **Debuggability**: Can inspect SQLite directly to verify data
4. **Simplicity**: Clear ownership of data (SQLite owns books, indices are caches)

### Why UUID Preservation Matters

Consider this scenario without UUID preservation:

```
Day 1: Import book from Google Books
       - UUID-A assigned, indexed in BM25 and FAISS

Day 2: Re-import same book (updated metadata)
       - UUID-B assigned (new UUID)
       - SQLite updated with UUID-B
       - Indices still reference UUID-A
       - Search finds the book, but hydration fails!
```

With UUID preservation:

```
Day 1: Import book from Google Books
       - UUID-A assigned, indexed

Day 2: Re-import same book
       - UUID-A preserved (existing)
       - Content updated in SQLite
       - Indices still valid (UUID-A exists)
       - System remains consistent
```

## Consequences

### Positive

1. **Rebuildable indices**: Can recover from index corruption
2. **Clear data ownership**: SQLite is authoritative
3. **Simple operations**: CRUD via repository pattern
4. **Portable**: Single SQLite file contains all book data
5. **Testable**: In-memory SQLite for fast integration tests

### Negative

1. **No real-time sync**: Index updates require explicit rebuild
   - Mitigation: Batch ingestion pattern; rebuild indices after import

2. **Limited query flexibility**: Denormalized schema limits ad-hoc queries
   - Mitigation: Sufficient for search-based access patterns

3. **JSON parsing overhead**: Authors/categories require JSON decode
   - Mitigation: Negligible for moderate catalog sizes

### Design Trade-offs

**Chose simplicity over normalization**: A normalized schema (separate authors, categories tables) would be more "correct" but adds complexity. For a thesis project focused on hybrid search, the denormalized approach is more appropriate.

**Chose consistency over performance**: UUID preservation adds a query to check existing rows on every save. This slight overhead ensures indices remain valid, which is more important than raw insert speed.

**Chose file-based over in-memory**: SQLite file persists across restarts. For a production system with very high throughput, an in-memory database with periodic snapshots might be faster, but SQLite files are simpler and sufficient.

## Implementation Notes

### File Location

```
app/infrastructure/db/sqlite_book_catalog_repository.py
```

### Key Classes

- `BookRow`: SQLAlchemy ORM model for the `books` table
- `SqliteBookCatalogRepository`: Implements `BookCatalogRepository` port

### Usage Example

```python
# Initialize
repo = SqliteBookCatalogRepository("sqlite:///data/catalog.db")

# Save (upsert)
book = Book.create_new(title="...", authors=["..."], source="google_books", source_id="abc123")
repo.save(book)

# Retrieve
found = repo.get_by_id(book.id)
found = repo.get_by_source_id("google_books", "abc123")

# List all
all_books = repo.get_all()
count = repo.count()

# Delete
repo.delete(book.id)
```

### Rebuilding Indices from SQLite

```python
# After ingestion or if indices are corrupted:
all_books = catalog_repo.get_all()

# Rebuild BM25
bm25_repo.build_index(all_books)

# Rebuild FAISS
for book in all_books:
    embedding = embeddings_store.generate_embedding(book.get_searchable_text())
    embeddings_store.store_embedding(book.id, embedding)
embeddings_store.build_index()
```

## References

- SQLAlchemy 2.0 Documentation: https://docs.sqlalchemy.org/en/20/
- SQLite Documentation: https://sqlite.org/docs.html
- ADR 002: BM25 for Lexical Search (derived from SQLite)
- ADR 003: FAISS for Vector Search (derived from SQLite)

## Related ADRs

- **ADR 002**: BM25 lexical search - index is derived from SQLite
- **ADR 003**: FAISS vector search - index is derived from SQLite
