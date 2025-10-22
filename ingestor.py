import uuid
from datetime import datetime
from typing import Any, Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

import schemas
from config import CONFIG


def ingest_jobs_to_rag(parsed_jobs: List[Dict[str, Any]], collection_name):
    """
    Ingest job listings into a ChromaDB collection with embeddings.

    Args:
        parsed_jobs: List of job dictionaries with keys like title, region, published, link, tags, description
    """
    # Initialize the sentence transformer model
    model = SentenceTransformer(CONFIG["embedding_model"])

    # Initialize ChromaDB client and collection
    client = chromadb.PersistentClient(path=CONFIG["chroma_db_path"])
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # Ensure cosine similarity
    )

    # Get existing source URLs to avoid duplicates
    existing_items = collection.get(include=["metadatas"])
    existing_urls = {
        item.get("source_url")
        for item in existing_items["metadatas"]
        if item and "source_url" in item
    }

    # Process jobs
    jobs_to_add = []
    embeddings_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    for job in parsed_jobs:
        source_url = job.get("link", "")

        # Skip if duplicate
        if source_url in existing_urls:
            continue

        # Clean description
        description = job.get("description", "")

        # Extract company if not provided
        company = job.get("company", "")

        tags_str = job.get("tags", [])

        # Prepare metadata
        metadata = {
            "job_id": str(uuid.uuid4()),
            "title": job.get("title", ""),
            "company": company,
            "location": job.get("region", ""),
            "description": description,
            "source_url": source_url,
            "posted_date": job.get("published", ""),
            "tags": tags_str,
            "ingested_at": datetime.utcnow().isoformat(),
        }
        metadata = schemas.JobMetadata(**metadata).model_dump()

        embedding_text = f"{job.get('title', '')} — {company} — {description}"
        embedding = model.encode(embedding_text).tolist()

        jobs_to_add.append(metadata)
        embeddings_to_add.append(embedding)
        metadatas_to_add.append(metadata)
        ids_to_add.append(metadata["job_id"])

    if ids_to_add:
        collection.add(
            embeddings=embeddings_to_add, metadatas=metadatas_to_add, ids=ids_to_add
        )
        print(f"Added {len(ids_to_add)} new jobs to the collection.")
    else:
        print("No new jobs to add (all duplicates).")


def clear_job_database(collection_name):
    """
    Delete all jobs from the 'remote_jobs' collection.
    """
    db_path = CONFIG["chroma_db_path"]
    if db_path is None:
        return

    client = chromadb.PersistentClient(path=db_path)

    # Check if collection exists
    existing_collections = [col.name for col in client.list_collections()]
    if collection_name not in existing_collections:
        print(f"Collection '{collection_name}' does not exist. Nothing to clear.")
        return
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name=collection_name)
    
    # Get all IDs in the collection
    all_results = collection.get(include=[])
    all_ids = all_results['ids']
    
    if all_ids:
        # Delete all documents
        collection.delete(ids=all_ids)
        print(f"Deleted {len(all_ids)} jobs from the collection.")
    else:
        print("Collection is already empty.")