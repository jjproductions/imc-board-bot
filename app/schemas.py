from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any


class IngestResponse(BaseModel):
    collection: str
    chunks_inserted: int
    points_upserted: int
    source_id: str


class DoclingBlock(BaseModel):
    id: str
    type: Literal[
        "heading",
        "paragraph",
        "list_item",
        "table",
        "figure",
        "caption",
        "footnote",
        "page_break",
    ] = "paragraph"
    text: Optional[str] = None
    page: Optional[int] = None
    level: Optional[int] = None  # heading level (1..6), if type == heading
    parent_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None  # table structure, figure info, etc.
    # Provenance and original OCR fields (from Docling debug output)
    prov: Optional[List[Dict[str, Any]]] = None
    orig: Optional[str] = None
    enumerated: Optional[bool] = None
    marker: Optional[str] = None
    content_layer: Optional[str] = None


class DoclingDocument(BaseModel):
    source_id: str = Field(
        ..., description="Unique ID of the document (e.g., path or SHA)."
    )
    title: Optional[str] = None
    blocks: List[DoclingBlock]


class ChunkBase(BaseModel):
    name: str
    description: Optional[str] = None


class ChunkCreate(ChunkBase):
    pass


class Chunk(ChunkBase):
    id: int
