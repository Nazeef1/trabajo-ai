"""
Vector store layer (RAG component).

Why this exists rather than just feeding the whole resume into the prompt:
when a resume is long, or when we eventually scale this to multiple resumes,
we don't want to stuff everything into context. Instead, the resume is
chunked into semantic sections (experience bullets, projects, skills, etc.)
and embedded. For each JD requirement, we retrieve only the most relevant
chunks before asking the LLM to reason about the match. This keeps the
reasoning grounded in retrieved evidence rather than the model's full,
unfiltered view of the resume — which is the actual point of RAG.

Uses ChromaDB's default embedding function (ONNX MiniLM under the hood,
bundled with chromadb) — no torch / sentence-transformers dependency needed.
"""
import re
import uuid
import chromadb


def chunk_resume(raw_text: str) -> list[str]:
    """
    Split resume text into semantically meaningful chunks.
    Splits on blank lines / bullet boundaries, then merges very short
    fragments so chunks roughly correspond to bullet points or short
    paragraphs (resume sections), not single words.
    """
    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", raw_text)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    chunks = []
    buffer = ""
    for line in lines:
        # Treat lines starting with bullet-like characters as new chunk boundaries
        is_bullet = bool(re.match(r"^[•\-\*\u2022]\s+", line))
        if is_bullet and buffer:
            chunks.append(buffer.strip())
            buffer = line
        else:
            buffer = f"{buffer} {line}".strip() if buffer else line
        # Cap chunk size so none get too long
        if len(buffer) > 400:
            chunks.append(buffer.strip())
            buffer = ""
    if buffer:
        chunks.append(buffer.strip())

    # Merge very short chunks (e.g. headers) into the next chunk
    merged = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if len(c) < 30 and i + 1 < len(chunks):
            merged.append(f"{c} {chunks[i+1]}")
            i += 2
        else:
            merged.append(c)
            i += 1
    return merged if merged else [raw_text]


class ResumeVectorStore:
    """
    Wraps an in-memory ChromaDB collection scoped to a single resume's
    chunks for the duration of one analysis request.
    """

    def __init__(self):
        # Ephemeral client: fine for our use case (one request = one session,
        # no need to persist across server restarts for the demo).
        self.client = chromadb.EphemeralClient()
        self.collection_name = f"resume_{uuid.uuid4().hex[:8]}"
        self.collection = self.client.create_collection(name=self.collection_name)
        self._loaded = False

    def index_resume(self, raw_text: str) -> int:
        """Chunk and embed the resume text. Returns number of chunks indexed."""
        chunks = chunk_resume(raw_text)
        if not chunks:
            return 0
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, ids=ids)
        self._loaded = True
        return len(chunks)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Retrieve the top-k resume chunks most relevant to a query
        (typically a JD requirement or skill)."""
        if not self._loaded:
            return []
        results = self.collection.query(query_texts=[query], n_results=k)
        docs = results.get("documents", [[]])
        return docs[0] if docs else []

    def cleanup(self):
        """Remove the collection once the request is done."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
