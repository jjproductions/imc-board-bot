from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
from app import crud, schemas
from ..settings import settings
from ..schemas import IngestResponse, DoclingDocument
from ..utilities import embed_texts, chunk_blocks, batched
import logging
import uuid
from datetime import datetime, timezone
from qdrant_client.models import PointStruct as QdrantPointStruct
import tempfile
import os
import json

router = APIRouter()

# qdrant-client may not be available in all environments (editor/CI); import at runtime
# and surface a clear error message if missing.
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        Filter,
        FieldCondition,
        MatchValue,
    )
except Exception as e:
    QdrantClient = None
    Distance = None
    VectorParams = None
    QdrantPointStruct = None
    Filter = None
    FieldCondition = None
    MatchValue = None
    _qdrant_import_err = e

if QdrantClient is None:
    raise RuntimeError(
        "Missing dependency 'qdrant-client'. Install with: pip install qdrant-client and ensure it's available in your environment."
    ) from _qdrant_import_err
qdrant = QdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key)

# Note: models/tokenizer are initialized lazily via `app.deps.get_*`.
# Avoid initializing heavy models at import time to prevent circular imports
# and to make tests/imports lightweight. Endpoints call `embed_texts` and
# `chunk_blocks`, which will initialize the resources when required.


def ensure_collection(collection_name: str, vector_dim: int):
    """
    Ensure the named Qdrant collection exists; create it with the provided vector_dim
    and cosine distance if it's missing.
    """
    try:
        existing = qdrant.get_collections().collections
    except Exception as e:
        raise RuntimeError(f"Failed to list qdrant collections: {e}")

    if any(col.name == collection_name for col in existing):
        # existing = qdrant.get_collection(collection_name)
        # If size mismatch, recreate for safety
        # size = existing.vectors_count or vector_dim
        # if hasattr(existing.config, "params") and existing.config.params and existing.config.params.vectors and existing.config.params.vectors.size != vector_dim:
        # return
        qdrant.delete_collection(collection_name)

    try:
        # The qdrant-client API expects the keyword `vectors_config` for newer versions.
        # Use `vectors_config` to be compatible with qdrant-client >=1.0.
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create qdrant collection '{collection_name}': {e}"
        )


# CRUD Endpoints for Chunks
@router.get("/chunks", response_model=List[schemas.Chunk])
def list_chunks():
    return crud.list_chunks()


@router.post("/chunks", response_model=schemas.Chunk, status_code=201)
def create_chunk(chunk: schemas.ChunkCreate):
    return crud.create_chunk(chunk)


@router.get("/chunks/{chunk_id}", response_model=schemas.Chunk)
def get_chunk(chunk_id: int):
    chunk = crud.get_chunk(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="chunk not found")
    return chunk


@router.put("/chunks/{chunk_id}", response_model=schemas.Chunk)
def update_chunk(chunk_id: int, chunk_in: schemas.ChunkCreate):
    chunk = crud.update_chunk(chunk_id, chunk_in)
    if not chunk:
        raise HTTPException(status_code=404, detail="chunk not found")
    return chunk


@router.delete("/chunks/{chunk_id}", status_code=204)
def delete_chunk(chunk_id: int):
    removed = crud.delete_chunk(chunk_id)
    if not removed:
        raise HTTPException(status_code=404, detail="chunk not found")
    return None


@router.get("/health")
def health():
    return {
        "status": "ok",
        "qdrant_url": settings.qdrant.url,
        "collection_default": settings.qdrant.default_collection,
        "embedding_model": settings.vector.embedding_model,
        "embedding_dim": settings.vector.embedding_dim,
    }


@router.post("/create-collection")
def create_collection(collection_name: str, vector_dim: Optional[int] = None):
    ensure_collection(collection_name, vector_dim or settings.vector.embedding_dim)
    return {
        "collection": collection_name,
        "vector_dim": vector_dim or settings.vector.embedding_dim,
    }


@router.post("/ingest", response_model=IngestResponse)
def ingest_doc(
    doc: DoclingDocument,
    collection: Optional[str] = None,
    max_tokens: int = settings.chunking.max_tokens,
    overlap_tokens: int = settings.chunking.overlap_tokens,
    batch_size: int = settings.chunking.batch_size,
    file_name: Optional[str] = None,
):
    """
    Ingest a Docling-parsed document:
    - chunks with heading-aware context and overlap
    - embeds via BAAI/bge-m3 (1024-dim, normalized)
    - upserts into Qdrant with rich payloads
    """
    coll = collection or settings.qdrant.default_collection
    ensure_collection(coll, settings.vector.embedding_dim)

    chunks = chunk_blocks(
        doc.blocks, max_tokens=max_tokens, overlap_tokens=overlap_tokens
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from document.")

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    points = []
    for idx, (c, vec) in enumerate(zip(chunks, vectors)):
        chunk_id = str(uuid.uuid4())
        payload = {
            "source_id": doc.source_id,
            "title": doc.title,
            "chunk_id": chunk_id,
            "chunk_index": idx,
            "text": c["text"],
            "tokens": c["tokens"],
            "pages": c["pages"],
            "section_path": c["context_path"],
            "block_ids": c["block_ids"],
            "overlap_from_previous": c["overlap_from_previous"],
            "embedding_model": settings.vector.embedding_model,
        }
        points.append(QdrantPointStruct(id=chunk_id, vector=vec, payload=payload))

    upserted = 0
    for batch in batched(points, batch_size):
        qdrant.upsert(collection_name=coll, points=batch)
        upserted += len(batch)

    return IngestResponse(
        collection=coll,
        chunks_inserted=len(chunks),
        points_upserted=upserted,
        source_id=doc.source_id,
    )


@router.post("/ingest-file", response_model=IngestResponse)
def ingest_file(
    file: UploadFile = File(...),
    collection: Optional[str] = Form(None),
    max_tokens: int = settings.chunking.max_tokens,
    overlap_tokens: int = settings.chunking.overlap_tokens,
    batch_size: int = settings.chunking.batch_size,
    source_id: Optional[str] = Form(None),
):
    """
    Accept an uploaded PDF, parse it with the external `docling` CLI into
    a flat JSON/document form, convert to `DoclingDocument` if necessary,
    then call the existing `ingest_doc` logic to chunk/embed/upsert.

    If the `docling` CLI is not available, this endpoint returns 501 and
    a short message explaining how to provide pre-parsed JSON instead.
    """
    logging.getLogger("app").info(
        f"Received file upload: {file.filename} ({file.content_type}), source_id={source_id}"
    )
    # Save uploaded file to a temp file
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)
    with open(tmp_path, "wb") as fh:
        fh.write(file.file.read())

    logging.getLogger("app").info(f"Saved uploaded file to temp path: {tmp_path}")
    # Prefer an HTTP Docling service if `DOCLING_URL` is configured (e.g. Docker)
    docling_url = settings.docling.url
    parsed = None
    if docling_url:
        # Use the Docling HTTP service (expects a /parse endpoint returning JSON)
        print(f"Using Docling HTTP service at {docling_url} to parse uploaded file...")
        try:
            import requests
        except Exception:
            os.unlink(tmp_path)
            raise HTTPException(
                status_code=500,
                detail="Missing dependency 'requests' required to call Docling HTTP service.",
            )

        parse_url = docling_url.rstrip("/") + "/v1/convert/file"
        logging.getLogger("app").info(
            f"Using Docling HTTP service at {parse_url} to parse uploaded file."
        )
        try:
            with open(tmp_path, "rb") as fh:
                files = {
                    "files": (
                        file.filename,
                        fh,
                        file.content_type or "application/pdf",
                    ),
                    "options": (
                        None,
                        json.dumps(
                            {
                                "to_formats": ["json"],  # list form is also accepted
                                "pdf_backend": "dlparse_v4",
                                "do_ocr": False,
                                "do_table_structure": True,
                                "table_mode": "accurate",
                            }
                        ),
                        "application/json",
                    ),
                }
                # Request Docling HTTP service using the flat profile
                resp = requests.post(parse_url, files=files, timeout=(5, 300))
        except Exception as e:
            os.unlink(tmp_path)
            raise HTTPException(
                status_code=502, detail=f"Failed to contact docling service: {e}"
            )

        # Clean up temp file
        os.unlink(tmp_path)

        if resp.status_code != 200:
            msg = (resp.text or "").strip()[:1000]
            raise HTTPException(
                status_code=500,
                detail=f"docling service error: {resp.status_code} {msg}",
            )

        try:
            parsed = resp.json()
            dump = parsed
            print(
                "Parsed JSON from Docling service:\n",
                json.dumps(dump.keys(), indent=2, ensure_ascii=False),
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"docling returned non-JSON: {e}"
            )
    else:
        # Fallback to invoking local `docling` CLI
        try:
            print("Use local `docling` package to parse uploaded file...")

            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                TableStructureOptions,
                AcceleratorOptions,
            )
            from docling.backend.docling_parse_v4_backend import (
                DoclingParseV4DocumentBackend,
            )

            # from quackling.llama_index.readers import DoclingPDFReader

            source = tmp_path  # file path or URL
            pipeline_opts = PdfPipelineOptions(
                artifacts_path=str(settings.docling.artifact_path or ""),
                do_ocr=False,
                do_table_structure=True,  # use TableFormer to get structured tables
                table_structure_options=TableStructureOptions(
                    do_cell_matching=True,  # often good with dlparse_v4
                    # mode can be "fast" or "accurate" depending on your needs
                    # mode=TableFormerMode.ACCURATE
                ),
                accelerator_options=AcceleratorOptions(num_threads=4, device="auto"),
            )
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_opts,
                        backend=DoclingParseV4DocumentBackend,
                    )
                },
            )
            print("Invoking local docling converter...")
            proc = converter.convert(source)
            # Convert the returned Docling Document object into a plain dict
            # and also produce a JSON string for logging/return if desired.
            try:
                if hasattr(proc.document, "export_to_dict"):
                    parsed = proc.document.export_to_dict()
                # elif hasattr(proc, "as_dict"):
                #     parsed = proc.as_dict()
                # elif hasattr(proc, "save_as_json"):
                #     parsed = json.loads(proc.save_as_json())
                else:
                    raise RuntimeError(
                        "Unsupported Docling document object: no known export method"
                    )

                # Use FastAPI's jsonable_encoder to handle non-JSON-native types
                try:
                    from fastapi.encoders import jsonable_encoder

                    serializable = jsonable_encoder(parsed)
                except Exception:
                    # Fallback: assume proc_dict is already JSON-serializable
                    serializable = parsed

                # JSON string (useful for returning/storing raw JSON)
                proc_json = json.dumps(serializable, ensure_ascii=False)

                # Save debug JSON with a UTC timestamp to avoid clobbering
                debug_filename = (
                    f"{settings.data_dir}/docling_debug_output_"
                    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
                )
                proc.document.save_as_json(filename=debug_filename)
                logging.getLogger("app").info(
                    "Docling document keys: %s",
                    (
                        list(parsed.keys())
                        if isinstance(parsed, dict)
                        else str(type(parsed))
                    ),
                )
            except Exception as e:
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to convert local docling output: {e}",
                )

        except FileNotFoundError as e:
            os.unlink(tmp_path)
            raise HTTPException(
                status_code=501,
                detail=(
                    "`docling` CLI not found on the server. Either install `docling` "
                    "or set DOCLING_URL to point at a docling HTTP service."
                ),
            )
        except ValueError as e:
            logging.getLogger("app").exception("Invalid options")
            raise HTTPException(status_code=400, detail=f"Invalid options: {e}")
        except Exception as e:
            logging.getLogger("app").exception("Docling conversion failure")
            raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

        # Clean up temp file (we already converted the document to `parsed` above)
        os.unlink(tmp_path)
        logging.getLogger("app").info(f"Deleted temp file: {tmp_path}")

    # If parsed already matches DoclingDocument (has source_id & blocks), use it.
    if isinstance(parsed, dict) and parsed.get("source_id") and parsed.get("blocks"):
        # Normalize possible page information that may be embedded in block.meta
        # or under different keys produced by different docling versions/backends.
        print("Normalizing parsed blocks for page information...")
        for b in parsed.get("blocks", []):
            print("Parsed Block JSON:\n", json.dumps(b, indent=2))
            if b.get("page") is None:
                meta = b.get("meta") or {}
                # Common alternative keys where page might live
                for key in ("page", "page_number", "pageno", "pageNo"):
                    # direct key on block
                    if key in b and b.get(key) is not None:
                        try:
                            b["page"] = int(b.get(key))
                            break
                        except Exception:
                            pass
                    # inside meta
                    if key in meta and meta.get(key) is not None:
                        try:
                            b["page"] = int(meta.get(key))
                            break
                        except Exception:
                            pass

        doc_obj = DoclingDocument.model_validate(parsed)
    # Support legacy (flat) Docling JSON that exposes `texts`/`tables` lists
    elif isinstance(parsed, dict) and parsed.get("texts"):
        logging.getLogger("app").info(
            "Converting parsed output to DoclingDocument from flat texts/tables..."
        )
        import re

        texts = parsed.get("texts", [])
        tables = parsed.get("tables", [])
        blocks = []

        # Convert text fragments -> blocks
        for i, t in enumerate(texts):
            raw = (t.get("text") or t.get("orig") or "").strip()
            if not raw:
                continue

            # page extraction from prov (if present)
            page = None
            prov = t.get("prov") or []
            if prov and isinstance(prov, list):
                p0 = prov[0]
                if isinstance(p0, dict) and p0.get("page_no") is not None:
                    try:
                        page = int(p0.get("page_no"))
                    except Exception:
                        page = None

            label = (t.get("label") or "").lower()
            btype = "paragraph"
            level = None

            # simple heuristics for headings
            if label in ("title", "heading", "formula", "section_header") or re.match(
                r"^(ARTICLE|SECTION)\b", raw, flags=re.I
            ):
                btype = "heading"
                if re.match(r"^ARTICLE\b", raw, flags=re.I):
                    level = 1
                elif re.match(r"^SECTION\b", raw, flags=re.I):
                    level = 2
            elif label in ("list_item",):
                btype = "list_item"
            elif label in ("caption",):
                btype = "caption"
            elif label in ("text"):
                btype = "paragraph"
            elif label in ("page_footer", "page_header"):
                btype = "page_break"

            parent = t.get("parent") or {}
            parent_label = parent.get("$ref") or ""

            blocks.append(
                {
                    "id": f"b{i+1}",
                    "type": btype,
                    "text": raw,
                    "page": page,
                    "level": level,
                    "meta": {"label": label},
                    "parent_id": parent_label,
                    "prov": prov,
                    "orig": t.get("orig"),
                    "enumerated": t.get("enumerated"),
                    "marker": t.get("marker"),
                    "content_layer": t.get("content_layer"),
                }
            )

        # Convert tables -> table blocks (basic markdown/grid fallback)
        start_idx = len(blocks)
        for j, tab in enumerate(tables, start=1):
            grid = tab.get("data", {}).get("grid") or []
            md_lines = []
            if grid:
                for row in grid:
                    cells = [cell.get("text", "").strip() for cell in row]
                    md_lines.append("| " + " | ".join(cells) + " |")
            else:
                cells = [
                    c.get("text", "").strip()
                    for c in tab.get("data", {}).get("table_cells", [])
                ]
                if cells:
                    md_lines.append(" | ".join(cells))

            table_text = "\n".join(md_lines) if md_lines else None

            page = None
            prov = tab.get("prov") or []
            if prov and isinstance(prov, list) and prov[0].get("page_no") is not None:
                try:
                    page = int(prov[0].get("page_no"))
                except Exception:
                    page = None

            blocks.append(
                {
                    "id": f"b{start_idx + j}",
                    "type": "table",
                    "text": table_text,
                    "page": page,
                    "level": None,
                    "meta": {"table": tab.get("data", {})},
                }
            )

        path_to_doc = (
            parsed.get("origin").get("filename") if parsed.get("origin") else None
        )
        path_to_repo = (
            settings.document_repository_path.removesuffix("/")
            if settings.document_repository_path
            else ""
        )
        if path_to_repo:
            # Prefix with your document repository path if available
            path_to_doc = path_to_repo.removesuffix("/") + "/" + path_to_doc

        # Determine source_id and title
        source_id_val = path_to_doc or source_id or file.filename
        title_val = parsed.get("name") or parsed.get("title") or file.filename

        norm_doc = {"source_id": source_id_val, "title": title_val, "blocks": blocks}
        doc_obj = DoclingDocument.model_validate(norm_doc)
        # Save debug JSON with a UTC timestamp to avoid clobbering
        debug_filename = (
            f"{settings.data_dir}/docling_converted_output_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        # Save converted DoclingDocument as JSON for debugging
        try:
            with open(debug_filename, "w", encoding="utf-8") as fh:
                fh.write(doc_obj.model_dump_json(indent=2))
            logging.getLogger("app").info(
                "Saved converted DoclingDocument to: %s", debug_filename
            )
        except Exception:
            logging.getLogger("app").exception(
                "Failed saving converted DoclingDocument to %s", debug_filename
            )
    else:
        # Fallback conversion: expect a `document` object with `md_content` (markdown)
        print("Converting parsed output to DoclingDocument from markdown content...")
        info = parsed.get("document", {}) if isinstance(parsed, dict) else {}
        md = info.get("md_content") or info.get("text") or info.get("content") or ""
        title = source_id or info.get("filename") or file.filename
        # Simple heading-aware split (reuse chunking expectations from ingestion flow)
        # We'll split headings (Markdown) and paragraphs into blocks.
        import re

        pattern = re.compile(r"^(?P<header>#{1,6})\s*(?P<title>.+)$", re.MULTILINE)
        blocks = []
        idc = 1
        matches = list(pattern.finditer(md))
        if not matches:
            paras = [p.strip() for p in re.split(r"\n\s*\n", md) if p.strip()]
            for p in paras:
                blocks.append({"id": f"b{idc}", "type": "paragraph", "text": p})
                idc += 1
        else:
            if matches and matches[0].start() > 0:
                intro = md[: matches[0].start()].strip()
                for p in [
                    pp.strip() for pp in re.split(r"\n\s*\n", intro) if pp.strip()
                ]:
                    blocks.append({"id": f"b{idc}", "type": "paragraph", "text": p})
                    idc += 1
            for i, m in enumerate(matches):
                header = m.group("header")
                title_text = m.group("title").strip()
                blocks.append(
                    {
                        "id": f"b{idc}",
                        "type": "heading",
                        "text": title_text,
                        "level": len(header),
                    }
                )
                idc += 1
                endpos = matches[i + 1].start() if i + 1 < len(matches) else len(md)
                body = md[m.end() : endpos].strip()
                if body:
                    for p in [
                        pp.strip() for pp in re.split(r"\n\s*\n", body) if pp.strip()
                    ]:
                        blocks.append({"id": f"b{idc}", "type": "paragraph", "text": p})
                        idc += 1

        doc_obj = DoclingDocument.model_validate(
            {"source_id": title, "title": title, "blocks": blocks}
        )

    # Delegate to the existing ingest_doc logic to avoid duplication
    return ingest_doc(
        doc_obj,
        collection=collection,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        batch_size=batch_size,
        file_name=debug_filename,
    )
