from typing import List, Dict, Any, Optional, Tuple
import re
from .deps import get_tokenizer, get_embedder
from .schemas import DoclingBlock
import logging
from .settings import settings


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def table_to_markdown(meta: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Convert Docling table meta (if available) to Markdown.
    Expected structures may vary; we attempt a best-effort conversion if rows exist.
    """
    if not meta:
        return None
    rows = meta.get("rows") or meta.get("table_rows") or None
    if not rows or not isinstance(rows, list):
        return None
    md_lines: List[str] = []
    header = rows[0] if rows else []
    header_cells = [str(c) for c in header]
    md_lines.append("| " + " | ".join(header_cells) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
    for row in rows[1:]:
        md_lines.append("| " + " | ".join([str(c) for c in row]) + " |")
    return "\n".join(md_lines)


def split_long_text_by_tokens(text: str, max_tokens: int) -> List[str]:
    """
    Token-aware splitting for very long paragraphs:
    - Try sentence splits
    - Assemble sentences into chunks not exceeding max_tokens
    """
    tokenizer = get_tokenizer()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sent in sentences:
        toks = tokenizer.encode(sent, add_special_tokens=False)
        if current_tokens + len(toks) <= max_tokens:
            current.append(sent)
            current_tokens += len(toks)
        else:
            if current:
                chunks.append(normalize_whitespace(" ".join(current)))
                current = []
                current_tokens = 0
            if len(toks) > max_tokens:
                # sentence itself is too long → hard split by tokens
                for i in range(0, len(toks), max_tokens):
                    sub_tokens = toks[i : i + max_tokens]
                    sub_text = tokenizer.decode(sub_tokens)
                    chunks.append(normalize_whitespace(sub_text))
            else:
                current.append(sent)
                current_tokens = len(toks)
    if current:
        chunks.append(normalize_whitespace(" ".join(current)))
    return chunks


def normalize_ocr_spacing(text: str) -> str:
    """
    Collapse spaced single-letter OCR artifacts like "O N E" -> "ONE".
    Conservative: only runs on sequences of uppercase letters separated by single spaces
    and of length >= 3 to avoid mangling initials or short codes.
    """
    if not text:
        return text

    def repl(m):
        s = m.group(0)
        return s.replace(" ", "")

    # Match sequences like "A B C" or "A B C D" (3+ letters)
    pattern = r"\b(?:[A-Z]\s){2,}[A-Z]\b"
    return re.sub(pattern, repl, text)


def chunk_blocks(
    blocks: List[DoclingBlock],
    max_tokens: int = settings.chunking.max_tokens,
    overlap_tokens: int = settings.chunking.overlap_tokens,
    source_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Heading-aware chunker (multilingual safe):
    - maintains a context stack for headings (level 1..6)
    - combines paragraphs/list/captions with that context
    - converts tables to markdown if possible
    - ensures token window with overlap
    Returns a list of chunk dicts: {text, tokens, pages, context_path, block_ids}
    """
    tokenizer = get_tokenizer()

    context_stack: List[Tuple[int, str]] = []  # (level, heading_text)
    buffer_texts: List[str] = []
    buffer_blocks: List[DoclingBlock] = []
    buffer_pages: List[int] = []
    buffer_tokens = 0
    chunks: List[Dict[str, Any]] = []

    def current_context_path() -> str:
        return " > ".join([h for _, h in context_stack])

    def flush_buffer():
        nonlocal buffer_texts, buffer_blocks, buffer_pages, buffer_tokens
        if not buffer_texts:
            return
        text = normalize_whitespace("\n\n".join(buffer_texts))
        if settings.chunking.ocr_normalize_spaced_letters:
            text = normalize_ocr_spacing(text)
        toks = tokenizer.encode(text, add_special_tokens=False)
        if len(toks) <= max_tokens:
            chunks.append(
                {
                    "text": text,
                    "tokens": len(toks),
                    "pages": sorted(set(buffer_pages)),
                    "context_path": current_context_path(),
                    "block_ids": [b.id for b in buffer_blocks],
                    "block_payloads": [b.model_dump() for b in buffer_blocks],
                    "prov": [p for b in buffer_blocks for p in (b.prov or [])],
                    "orig_texts": [b.orig for b in buffer_blocks if b.orig],
                    "enumeration": [
                        {"id": b.id, "enumerated": b.enumerated, "marker": b.marker}
                        for b in buffer_blocks
                    ],
                    "source_id": source_id,
                    "overlap_from_previous": 0,
                }
            )
        else:
            sub_texts = split_long_text_by_tokens(text, max_tokens)
            for idx, sub in enumerate(sub_texts):
                sub_toks = tokenizer.encode(sub, add_special_tokens=False)
                chunks.append(
                    {
                        "text": sub,
                        "tokens": len(sub_toks),
                        "pages": sorted(set(buffer_pages)),
                        "context_path": current_context_path(),
                        "block_ids": [b.id for b in buffer_blocks],
                        "block_payloads": [b.model_dump() for b in buffer_blocks],
                        "prov": [p for b in buffer_blocks for p in (b.prov or [])],
                        "orig_texts": [b.orig for b in buffer_blocks if b.orig],
                        "enumeration": [
                            {"id": b.id, "enumerated": b.enumerated, "marker": b.marker}
                            for b in buffer_blocks
                        ],
                        "source_id": source_id,
                        "overlap_from_previous": 0 if idx == 0 else overlap_tokens,
                    }
                )
        buffer_texts = []
        buffer_blocks = []
        buffer_pages = []
        buffer_tokens = 0

    def add_text_with_window(add_text: str, block: DoclingBlock, page: Optional[int]):
        nonlocal buffer_texts, buffer_blocks, buffer_pages, buffer_tokens
        if not add_text:
            return

        if settings.chunking.ocr_normalize_spaced_letters:
            add_text = normalize_ocr_spacing(add_text)

        add_tokens = tokenizer.encode(add_text, add_special_tokens=False)
        if len(add_tokens) > max_tokens:
            for sub in split_long_text_by_tokens(add_text, max_tokens):
                add_text_with_window(sub, block, page)
            return

        if buffer_tokens + len(add_tokens) > max_tokens and buffer_texts:
            buffer_text = normalize_whitespace("\n\n".join(buffer_texts))
            buffer_toks = tokenizer.encode(buffer_text, add_special_tokens=False)
            if len(buffer_toks) > overlap_tokens:
                start = max(0, len(buffer_toks) - overlap_tokens)
                overlap_text = tokenizer.decode(buffer_toks[start:])
            else:
                overlap_text = buffer_text

            chunks.append(
                {
                    "text": buffer_text,
                    "tokens": len(buffer_toks),
                    "pages": sorted(set(buffer_pages)),
                    "context_path": current_context_path(),
                    "block_ids": [b.id for b in buffer_blocks],
                    "block_payloads": [b.model_dump() for b in buffer_blocks],
                    "prov": [p for b in buffer_blocks for p in (b.prov or [])],
                    "orig_texts": [b.orig for b in buffer_blocks if b.orig],
                    "enumeration": [
                        {"id": b.id, "enumerated": b.enumerated, "marker": b.marker}
                        for b in buffer_blocks
                    ],
                    "source_id": source_id,
                    "overlap_from_previous": 0,
                }
            )

            buffer_texts = [normalize_whitespace(overlap_text)]
            buffer_blocks = []
            buffer_pages = []
            buffer_tokens = len(
                tokenizer.encode(buffer_texts[0], add_special_tokens=False)
            )

        buffer_texts.append(add_text)
        buffer_blocks.append(block)
        if page is not None:
            buffer_pages.append(page)
        buffer_tokens += len(add_tokens)

    logging.getLogger("app").info(
        f"Chunking {len(blocks)} blocks with max_tokens={max_tokens}, overlap_tokens={overlap_tokens}"
    )
    for b in blocks:
        if b.type == "heading":
            h_text = normalize_whitespace(b.text or "")
            if not h_text:
                continue
            # Flush any buffered content first so it remains associated
            # with the previous heading/context. Updating the context
            # before flushing caused previous text to be attached to
            # the new (following) heading.
            flush_buffer()
            level = b.level or 1
            context_stack = [h for h in context_stack if h[0] < level]
            context_stack.append((level, h_text))
        elif b.type in {"paragraph", "list_item", "caption", "footnote"}:
            # print("Block:", b.model_dump())
            # print("Block JSON:\n", b.model_dump_json(indent=2))
            # log.debug("Block: %s", b.model_dump())             # dumps via str(); safe but not pretty
            # log.debug("Block JSON:\n%s", b.model_dump_json(indent=2))  # full pretty json in logs
            add_text_with_window(normalize_whitespace(b.text or ""), b, b.page)
        elif b.type == "table":
            md = table_to_markdown(b.meta)
            if md:
                add_text_with_window(md, b, b.page)
            else:
                add_text_with_window(normalize_whitespace(b.text or ""), b, b.page)
        elif b.type == "figure":
            continue
        elif b.type == "page_break":
            flush_buffer()
        else:
            logging.getLogger("app").warning(f"Unknown block type: {b.type}")
            add_text_with_window(normalize_whitespace(b.text or ""), b, b.page)

    flush_buffer()
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    # bge-m3: normalize for cosine similarity
    embedder = get_embedder()
    vectors = embedder.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def batched(iterable, n: int):
    """Yield successive n-sized batches from iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch
