from app.utilities import chunk_blocks
from app.schemas import DoclingBlock


def test_heading_context_assignment():
    blocks = [
        {"id": "b1", "type": "heading", "text": "First Heading", "level": 1},
        {"id": "b2", "type": "paragraph", "text": "Intro text belongs to first"},
        {"id": "b3", "type": "heading", "text": "Second Heading", "level": 1},
        {"id": "b4", "type": "paragraph", "text": "Text belongs to second"},
    ]

    # Parse to DoclingBlock Pydantic models (utilities.expect that shape)
    parsed = [DoclingBlock.model_validate(b) for b in blocks]

    chunks = chunk_blocks(parsed, max_tokens=200)

    # Build a mapping from block_id -> context_path by looking at block_ids in chunks
    block_to_context = {}
    for c in chunks:
        for bid in c.get("block_ids", []):
            block_to_context[bid] = c.get("context_path")

    # The paragraph after the first heading should be associated with "First Heading"
    assert block_to_context.get("b2") == "First Heading"

    # The paragraph after the second heading should be associated with "Second Heading"
    assert block_to_context.get("b4") == "Second Heading"
