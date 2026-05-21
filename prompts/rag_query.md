# RAG Query Prompt

Use this template when constructing the user-facing query sent to the language model during retrieval-augmented generation.

---

## Template

```
You are answering a question about board policies using only the retrieved context below.

Context:
{context}

Question:
{question}

Answer concisely and cite the policy section where applicable.
If the answer is not in the context, say: "I don't have that information in the current policy documents."
```

---

## Variables

| Variable | Description |
|---|---|
| `{context}` | Concatenated retrieved chunk texts, formatted as `[Section: <path>]\n<text>` |
| `{question}` | The user's original question, passed through verbatim |

---

## Notes

- Keep the context window in mind — trim or summarize chunks if total token count approaches the model limit.
- For hybrid search results, chunks are ranked by RRF score; include the top-k (default: 5) chunks.
