"""
Vector store layer (RAG component).
Uses ChromaDB's default embedding function (ONNX MiniLM under the hood, bundled with chromadb).
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

    text = re.sub(r"\r\n", "\n", raw_text)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    chunks = []
    buffer = ""
    for line in lines:
        is_bullet = bool(re.match(r"^[•\-\*\u2022]\s+", line))
        if is_bullet and buffer:
            chunks.append(buffer.strip())
            buffer = line
        else:
            buffer = f"{buffer} {line}".strip() if buffer else line
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
