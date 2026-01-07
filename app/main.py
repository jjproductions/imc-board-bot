from fastapi import FastAPI
from .routes.api import router as api_router
from .settings import settings

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title=settings.app_name)

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
