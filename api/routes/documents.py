"""Document Ingestion & Catalog Routes for DocMind API."""

import os
import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from config import settings
from api.schemas import DocumentUploadResponse, DocumentListResponse, DocumentMetadataItem
from api.dependencies import AppState, get_app_state
from ingestion.pipeline import IngestionPipeline
from vectorstore.store import get_or_create_faiss, replace_document_vectors

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {".pdf", ".csv", ".txt", ".md", ".markdown"}


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    state: AppState = Depends(get_app_state),
):
    """Uploads and ingests a document into DocMind's vector knowledge base."""
    filename = file.filename or "uploaded_doc"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {list(ALLOWED_EXTENSIONS)}",
        )

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / filename

    # Save uploaded file
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write uploaded file to disk: {str(e)}",
        )

    # Ingest document
    try:
        pipeline = IngestionPipeline(
            chunk_size=settings.default_chunk_size,
            chunk_overlap=settings.default_chunk_overlap,
        )
        chunks, report = pipeline.run(str(destination))

        # Check if existing index was reused
        if report.is_reused:
            entry = pipeline.registry.get_entry(report.content_fingerprint or "")
            doc_id = entry.doc_id if entry else "doc_cached"
            chunks_count = entry.chunks_count if entry else report.final_chunks_count
            total_chars = entry.character_count if entry else 0
            return DocumentUploadResponse(
                message="Document already indexed. Reusing existing vector index without re-processing.",
                filename=filename,
                file_type=ext.lstrip("."),
                chunks_created=chunks_count,
                doc_id=doc_id,
                character_count=total_chars,
                is_reused=True,
                content_fingerprint=report.content_fingerprint,
                config_signature=report.config_signature,
            )

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="File was parsed but produced zero text chunks.",
            )

        # Update Vector Store using Build-New-Then-Swap failure-safe atomic replacement
        if state.vectorstore is None:
            state.vectorstore = get_or_create_faiss(documents=chunks, embeddings=state.embedder)
        else:
            if report.content_fingerprint:
                replace_document_vectors(
                    vectorstore=state.vectorstore,
                    fingerprint_or_doc_id=report.content_fingerprint,
                    new_documents=chunks,
                )
            else:
                state.vectorstore.add_documents(chunks)

        total_chars = sum(len(c.page_content) for c in chunks)
        doc_id = chunks[0].metadata.get("parent_doc_id", "doc_unknown")

        return DocumentUploadResponse(
            message="Document successfully ingested and indexed.",
            filename=filename,
            file_type=ext.lstrip("."),
            chunks_created=len(chunks),
            doc_id=doc_id,
            character_count=total_chars,
            is_reused=False,
            content_fingerprint=report.content_fingerprint,
            config_signature=report.config_signature,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """Lists all uploaded documents and catalog items."""
    data_dir = Path("data")
    if not data_dir.exists():
        return DocumentListResponse(total_documents=0, documents=[])

    files = [f for f in data_dir.iterdir() if f.is_file()]
    items: List[DocumentMetadataItem] = []

    for f in sorted(files, key=lambda x: x.name):
        ext = f.suffix.lstrip(".").lower()
        size_bytes = f.stat().st_size
        items.append(
            DocumentMetadataItem(
                filename=f.name,
                file_type=ext,
                size_bytes=size_bytes,
                chunks_count=0,
                doc_id=None,
            )
        )

    return DocumentListResponse(
        total_documents=len(items),
        documents=items,
    )
