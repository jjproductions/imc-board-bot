from fastapi import FastAPI
from .routes.api import router as api_router
from .settings import settings
import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()

# Use the Lifespan context manager instead of the deprecated @app.on_event
from .deps import init_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan: initialize heavy resources and optionally start debug listener.

    This runs once when the application starts and yields control to the app.
    Using lifespan avoids the deprecated `@app.on_event("startup")` API.
    """
    # Non-blocking debugpy listener for local development (optional)
    try:
        import debugpy  # type: ignore
    except Exception:
        debugpy = None

    if debugpy is not None:
        try:
            debugpy.listen(("0.0.0.0", 5678))
            logging.getLogger("app").info("debugpy listening on 0.0.0.0:5678 (attach with VS Code)")
        except Exception as e:
            logging.getLogger("app").exception("Failed to start debugpy listener: %s", e)

    # Initialize models/tokenizer now so errors surface on startup
    try:
        init_models()
        logging.getLogger("app").info("Models initialized")
    except Exception as e:
        logging.getLogger("app").exception("Model initialization failed: %s", e)
        # Re-raise to prevent app from starting in a broken state
        raise

    try:
        yield
    finally:
        # optional cleanup can go here
        pass


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title=settings.app_name, lifespan=lifespan)

# Routes
app.include_router(api_router, prefix="/api")


# class SearchRequest(BaseModel):
#     query: str
#     top_k: int = 5
#     collection: Optional[str] = None
#     filter_by_source_id: Optional[str] = None


# class SearchHit(BaseModel):
#     score: float
#     text: str
#     payload: Dict[str, Any]


# class SearchResponse(BaseModel):
#     query: str
#     hits: List[SearchHit]



# @app.post("/search", response_model=SearchResponse)
# def semantic_search(req: SearchRequest):
#     """
#     Semantic search over the collection:
#     - embeds the query using bge-m3
#     - optional filter by source_id
#     """
#     coll = req.collection or DEFAULT_COLLECTION
#     ensure_collection(coll, EMBEDDING_DIM)

#     query_vec = embed_texts([req.query])[0]
#     q_filter: Optional[Filter] = None
#     if req.filter_by_source_id:
#         q_filter = Filter(
#             must=[FieldCondition(
#                 key="source_id",
#                 match=MatchValue(value=req.filter_by_source_id)
#             )]
#         )

#     result = qdrant.search(
#         collection_name=coll,
#         query_vector=query_vec,
#         limit=req.top_k,
#         with_payload=True,
#         filter=q_filter
#     )

#     hits = []
#     for r in result:
#         hits.append(SearchHit(
#             score=r.score,
#             text=r.payload.get("text", ""),
#             payload=r.payload
#         ))

#     return SearchResponse(query=req.query, hits=hits)
