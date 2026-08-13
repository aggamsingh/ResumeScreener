import os
import sys
import uuid
import re
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

def resume_chunks(text: str, words_per_chunk: int = 220, overlap: int = 40) -> list[str]:
    words = re.sub(r"\s+", " ", text or "").strip().split(" ")
    if not words or not words[0]:
        return []
    step = max(1, words_per_chunk - overlap)
    return [" ".join(words[start : start + words_per_chunk]) for start in range(0, len(words), step)]

def main():
    print("[Standalone Indexer] Initializing PyTorch model all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("[Standalone Indexer] Connecting to embedded Qdrant DB at ./data/qdrant_db...")
    client = QdrantClient(path="./data/qdrant_db")
    collection_name = "resumes"

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    dim = model.get_sentence_embedding_dimension()
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:root@127.0.0.1:5432/resume_lens")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    cursor.execute("SELECT id, full_name, email, skills, years_experience, location, resume_text, source_file_url FROM resumes WHERE resume_text IS NOT NULL AND length(trim(resume_text)) > 0")
    rows = cursor.fetchall()
    total = len(rows)
    print(f"[Standalone Indexer] Found {total} resumes to embed into Qdrant...")

    batch_points = []
    processed = 0

    for candidate_id, name, email, skills, years, location, resume_text, source_url in rows:
        try:
            raw_chunks = resume_chunks(resume_text)
            if not raw_chunks:
                continue
            skills_str = ", ".join(skills) if (skills and isinstance(skills, list)) else "N/A"
            summary_chunk = f"QUALIFICATION SUMMARY | Candidate: {name or 'Candidate'} | Experience: {years or 0} years | Location: {location or 'N/A'} | Core Skills & Tech Stack: {skills_str} | Resume: {(resume_text or '')[:1200]}"
            chunks = [summary_chunk] + raw_chunks
            vectors = model.encode(chunks, batch_size=128, show_progress_bar=False).tolist()
            payload = {
                "candidate_id": str(candidate_id), "name": name or "Unnamed candidate", "cv_path": source_url,
                "experience_years": years, "location": location, "location_raw": location,
                "skills": skills or [], "email": email, "ocr_used": False,
            }
            points = [PointStruct(id=str(uuid.uuid4()), vector=vector, payload={**payload, "chunk_text": chunk}) for chunk, vector in zip(chunks, vectors)]
            batch_points.extend(points)
            processed += 1

            if len(batch_points) >= 500:
                client.upsert(collection_name=collection_name, points=batch_points, wait=False)
                batch_points = []

            if processed % 1000 == 0:
                print(f"[Standalone Indexer] Progress: {processed}/{total} resumes embedded")
        except Exception as exc:
            print(f"[Standalone Indexer Warning] Skipped {candidate_id}: {exc}")

    if batch_points:
        client.upsert(collection_name=collection_name, points=batch_points, wait=True)

    cursor.close()
    conn.close()
    print(f"[Standalone Indexer] COMPLETED SUCCESSFULLY! {processed}/{total} resumes indexed.")

if __name__ == "__main__":
    main()
