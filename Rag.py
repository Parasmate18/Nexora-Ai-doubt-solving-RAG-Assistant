
# ============================================================
# PART 1 — IMPORTS AND ENVIRONMENT
# ============================================================

import os
import json
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType
)
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# PART 2 — CONNECT TO QDRANT
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=120
)

groq_client = Groq(api_key=GROQ_API_KEY)

print("Connected to Qdrant Cloud!")


# ============================================================
# PART 3 — CREATE QDRANT COLLECTION
# ============================================================

COLLECTION_NAME = "knowledge_filter"
EMBEDDING_SIZE = 384

if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE
    )
)

print("Collection created successfully!")


# ============================================================
# PART 4 — CREATE PAYLOAD INDEXES
# ============================================================

for field in ["category", "subcategory"]:
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=field,
        field_schema=PayloadSchemaType.KEYWORD
    )

print("Payload indexes created!")

documents = []

TRANSCRIPT_FOLDER = Path(__file__).resolve().parent / "transcripts"

for file_path in TRANSCRIPT_FOLDER.glob("*.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        documents.append(json.load(f))

print("Total transcripts:", len(documents))

# ============================================================
# PART 6 — LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model ready!")


# ============================================================
# PART 7 — FLATTEN SEGMENTS AND PREPARE METADATA
# ============================================================

texts = []
metadata = []

for document in documents:

    required_fields = [
        "video_id",
        "title",
        "category",
        "subcategory",
        "segments"
    ]

    for field in required_fields:
        if field not in document:
            raise ValueError(
                f"Missing '{field}' in transcript: {document.get('title', 'Unknown')}"
            )

    for segment in document["segments"]:

        text = segment["text"].strip()

        if not text:
            continue

        texts.append(text)

        metadata.append({
            "text": text,
            "video_id": document["video_id"],
            "title": document["title"],
            "category": document["category"],
            "subcategory": document["subcategory"],
            "start": segment["start"],
            "end": segment["end"]
        })

print(f"Total transcript segments: {len(texts)}")

from collections import Counter

print("Category distribution:",
      Counter(item["category"] for item in metadata))

print("Subcategory distribution:",
      Counter(item["subcategory"] for item in metadata))


# ============================================================
# PART 8 — GENERATE EMBEDDINGS
# ============================================================

embeddings = model.encode(texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")



# ============================================================
# PART 9 — CREATE AND INSERT QDRANT POINTS IN BATCHES
# ============================================================

BATCH_SIZE = 50

points = []

for i in range(len(texts)):
    point = PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload=metadata[i]
    )
    points.append(point)

print(f"Uploading {len(points)} points to Qdrant...")

for start in range(0, len(points), BATCH_SIZE):
    batch = points[start:start + BATCH_SIZE]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch,
        wait=True,
        timeout=120
    )

    print(f"Uploaded {min(start + BATCH_SIZE, len(points))}/{len(points)}")

print("All transcript segments uploaded successfully!")



# ============================================================
# PART 10 — SEARCH WITH CATEGORY FILTER
# ============================================================

def search_with_filter(query, category=None, subcategory=None, top_k=3):

    query_vector = model.encode(query).tolist()

    conditions = []

    if category:
        conditions.append(
            FieldCondition(
                key="category",
                match=MatchValue(value=category)
            )
        )

    if subcategory:
        conditions.append(
            FieldCondition(
                key="subcategory",
                match=MatchValue(value=subcategory)
            )
        )

    search_filter = Filter(must=conditions) if conditions else None

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True
    ).points

    return results


# ============================================================
# PART 11 — PREPARE CONTEXT
# ============================================================

def prepare_context(results):

    context_parts = []

    for result in results:

        payload = result.payload

        start = payload["start"]
        end = payload["end"]

        start_time = f"{int(start // 60):02d}:{int(start % 60):02d}"
        end_time = f"{int(end // 60):02d}:{int(end % 60):02d}"

        context_parts.append(
            f"""
Video Title: {payload['title']}
Category: {payload['category']}
Subcategory: {payload['subcategory']}
Timestamp: {start_time} - {end_time}
Video ID: {payload['video_id']}
Content: {payload['text']}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# PART 12 — ASK GROQ LLM
# ============================================================

def ask_llm(question, context):

    prompt = f"""
You are NEXORA AI, an expert Web Development learning assistant specializing in HTML and CSS.

Answer the question using only the provided video context.

Instructions:
1. Explain the concept clearly and simply.
2. Do not make up information.
3. Mention the video title.
4. Always provide relevant timestamps.
5. If multiple videos explain the concept, mention their timestamps.
6. Include the video ID.
7. If the answer is unavailable, say:
"I don't know based on the provided information."

Context:
{context}

Question:
{question}
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# ============================================================
# PART 13 — TEST SEARCH AND GENERATE ANSWER
# ============================================================

question = "Where are css fonts explained?"

results = search_with_filter(
    query=question,
    category="CSS",
    top_k=3
)

print("\nSearch Results:")

for result in results:
    print(f"Score: {result.score:.3f}")
    print(result.payload["text"])
    print()

context = prepare_context(results)

answer = ask_llm(question, context)

print("\n========== FINAL ANSWER ==========")
print(answer)
