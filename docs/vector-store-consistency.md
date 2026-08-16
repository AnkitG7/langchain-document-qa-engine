# Vector Store Consistency

This document outlines the vector store consistency system in the DocMind engine. Our architecture handles data consistency, duplication, and staleness problems that occur during vector embedding ingestion, ensuring reliable retrieval for RAG.

## 1. The Duplicate Vector Problem
- **Problem**: Without deduplication, uploading the same PDF $N$ times creates $N \times chunks$ vectors. Top-K retrieval then returns identical chunks from different upload sessions, wasting the LLM's valuable context window with redundant information.
- **Solution**: Content-based fingerprinting prevents re-indexing. By identifying documents through their intrinsic byte-content rather than path or name, duplicate uploads are skipped entirely.

## 2. The Stale Vector Problem
- **Problem**: When configuration changes (e.g., modifying `chunk_size` from 500 to 1000), old vectors have different chunk boundaries. If old 500-char chunks are mixed with new 1000-char chunks in the same index, it creates inconsistencies. Furthermore, if the embedding model changes (e.g., from 768 dimensions to 1536), dimension mismatches occur, which can cause index crashes or return garbage similarity scores.
- **Solution**: We track a Configuration Signature that detects drift in ingestion parameters. Any drift automatically triggers vector replacement, ensuring the index stays aligned with the current schema and chunking strategies.

## 3. Vector Contamination
- **Problem**: If old vectors are not physically removed, the vector index contains a mixture of old and new representations. Even if a document registry claims a document is "re-indexed", the stale vectors still physically reside in the vectorstore and pollute retrieval results.
- **Solution**: Physical vector deletion. We explicitly invoke `delete_documents_by_fingerprint()` to prune stale vectors before (or after, in our Build-New-Then-Swap model) inserting new vectors into the active index.

## 4. Metadata Used for Identifying Vectors
Vectors belong to documents. To successfully trace vectors back to their parent documents during deletion, we inject standard metadata fields at ingestion. 

The primary metadata fields are:
- `content_fingerprint`: SHA-256 of the raw binary file bytes.
- `parent_doc_id`: Unique document identifier (e.g., `doc_<12-char-hash>`).
- `doc_id`: Document identifier alias.
- `filename`: Original filename.
- `source`: File path or source URI.

These fields are attached to each document chunk during the pipeline run:
- See [`pipeline.py`](file:///c:/FS/langchain_document_qa/ingestion/pipeline.py#L116-L121) for text pipelines.
- See [`multimodal_pipeline.py`](file:///c:/FS/langchain_document_qa/ingestion/multimodal_pipeline.py#L204-L208) for multimodal pipelines.

## 5. FAISS Implementation
The local FAISS implementation handles deletion via in-memory structures.

In [`vectorstore/store.py`](file:///c:/FS/langchain_document_qa/vectorstore/store.py#L241-L263), `delete_documents_by_fingerprint()`:
- Scans all entries in `vectorstore.docstore._dict`.
- Matches against ANY of 5 metadata fields (`content_fingerprint`, `parent_doc_id`, `doc_id`, `filename`, `source`).
- Respects `exclude_ids` to protect newly inserted vectors during a Build-New-Then-Swap update.
- Calls `vectorstore.delete(matching_ids)` with a fallback to direct docstore dictionary cleanup (`_dict.pop`) if deletion is not directly supported by the index.
- **Limitations**: Non-transactional (no ACID guarantees), single-process only.
- Batched initialization: Chunks are indexed in batches of 50 to prevent payload exhaustion during massive inserts.

## 6. Chroma Implementation
The Chroma vectorstore relies on metadata queries for deletion.

In [`vectorstore/store.py`](file:///c:/FS/langchain_document_qa/vectorstore/store.py#L264-L282):
- Uses `_collection.get(where={"$or": [{"content_fingerprint": ...}, {"parent_doc_id": ...}]})` metadata query to fetch matching vector IDs.
- Filters out any `exclude_ids` from the returned deletion list.
- Calls `_collection.delete(ids=ids_to_del)` to perform the removal.
- **Fallback**: If the `$or` query fails (due to driver limitations or bugs), it tries a simpler `where={"content_fingerprint": ...}` filter.
- **Limitations**: Eventual consistency model. Transient read windows may see both versions (stale and new) for milliseconds before the delete fully propagates.

## 7. PGVector Implementation
For production environments, PGVector handles vectors gracefully using standard database ACID principles.

In [`production/pgvector_store.py`](file:///c:/FS/langchain_document_qa/production/pgvector_store.py):
- **True ACID transaction atomicity** via PostgreSQL.
- Executes `BEGIN TRANSACTION` → `DELETE matching rows` → `INSERT new vectors` → `COMMIT`.
- **On failure**: Automatic `ROLLBACK` restores the previous state, ensuring zero data corruption.
- Full MVCC (Multi-Version Concurrency Control) allowing multi-worker concurrency with connection pooling.

## 8. Build-New-Then-Swap
To prevent data loss during ingestion drift or update, we use a Build-New-Then-Swap architecture. See `replace_document_vectors()` in [`vectorstore/store.py`](file:///c:/FS/langchain_document_qa/vectorstore/store.py#L286-L319).

### The BAD (Naive) Architecture
```text
DELETE old vectors
→ Generate new embeddings
→ FAILURE (API timeout / rate limit)
→ DOCUMENT HAS ZERO VECTORS (data loss!)
```

### The SAFE Architecture
```text
1. Generate new embeddings (vectorstore.add_documents)
2. Insert new vectors into index
3. On success: delete old vectors (exclude_ids=set(inserted_ids))
4. New version active
```
*On failure at step 1:*
→ Exception is raised.
→ No delete has occurred yet.
→ Old vectors remain 100% intact.
→ Queries continue serving the old version seamlessly.

### Failure Behavior Scenarios
- **Embedding API fails (e.g., Ollama 504)**: Exception at `add_documents`, old vectors remain untouched.
- **Embedding timeout**: Exception occurs before delete. Old vectors intact.
- **Rate limit (429)**: Exception occurs before delete. Old vectors intact.
- **Process crashes**: 
  - **FAISS**: In-memory index is lost entirely unless previously saved to disk.
  - **Chroma**: Persistent collections survive; in-memory collections are lost.
  - **PGVector**: ACID rollback preserves the previous committed state intact.
- **Database/network failure during delete step**: New vectors have already been inserted. The old vectors may partially remain (FAISS/Chroma contamination) or a clean rollback occurs (PGVector).

### Workflow Diagram

```mermaid
flowchart TD
    Start[Document Drift Detected] --> Embed[Generate New Embeddings]
    Embed -->|Success| Insert[Insert New Vectors]
    Embed -->|Failure| Abort[Abort. Old Vectors Safe]
    Insert --> Delete[Delete Old Vectors\nExclude newly inserted IDs]
    Delete -->|Success| Done[New Version Active]
    Delete -->|Failure| Partial[Contamination (FAISS/Chroma)\nor Rollback (PGVector)]
```

### The `exclude_ids` Parameter
When new vectors are inserted, they intrinsically share the same `content_fingerprint` as the old vectors. Without `exclude_ids`, the deletion step would blindly match the fingerprint and delete the NEW vectors alongside the old ones. The `exclude_ids` parameter protects the newly inserted vector IDs from being pruned.

### Backend Atomicity Comparison

| Backend | Atomicity Level | Concurrency | Failure Mode |
|---------|-----------------|-------------|--------------|
| **FAISS** | None | Single-process | Partial state (Index may corrupt if not saved) |
| **Chroma** | Eventual Consistency | Multi-process (Persistent) | Transient contamination |
| **PGVector** | Full ACID | Multi-worker (MVCC + Pooling) | Safe Rollback |
