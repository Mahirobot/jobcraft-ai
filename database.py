import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
from bs4 import BeautifulSoup
from dateutil import parser  # pip install python-dateutil
from pydantic import BaseModel, field_validator
from sentence_transformers import SentenceTransformer

from config import CONFIG
from schemas import JobMetadata
from scraper import clean_description, extract_company_name


def search_jobs(query: str, n_results: int = 5, collection_names: list = None):
    """
    Search for jobs across multiple ChromaDB collections using semantic similarity.

    Args:
        query: Search query string
        n_results: Total number of results to return across all collections
        collection_names: List of collection names to search. If None, searches all collections.
    """
    client = chromadb.PersistentClient(path=CONFIG["chroma_db_path"])

    # Determine which collections to search
    if collection_names is None:
        # Get all collection names in the database
        all_collections = client.list_collections()
        collection_names = [col.name for col in all_collections]
        print(f"Searching all collections: {collection_names}")
    else:
        print(f"Searching specified collections: {collection_names}")

    if not collection_names:
        print("No collections found or specified.")
        return {"metadatas": [], "documents": [], "distances": [], "ids": []}

    model = SentenceTransformer(CONFIG["embedding_model"])  # Use model from config
    query_embedding = model.encode(query).tolist()

    all_results = {
        "metadatas": [],
        "documents": [],
        "distances": [],
        "ids": [],
        "collection_names": [],  # To track which collection each result came from
    }

    # Search each specified collection
    for col_name in collection_names:
        try:
            collection = client.get_collection(name=col_name)
            # Query the individual collection
            col_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,  # Get top N from each collection
                include=["metadatas", "documents", "distances"],
            )

            # Append results from this collection
            all_results["metadatas"].extend(
                col_results["metadatas"][0] if col_results["metadatas"] else []
            )
            all_results["documents"].extend(
                col_results["documents"][0] if col_results["documents"] else []
            )
            all_results["distances"].extend(
                col_results["distances"][0] if col_results["distances"] else []
            )
            all_results["ids"].extend(
                col_results["ids"][0] if col_results["ids"] else []
            )
            # Add collection name for each result from this collection
            all_results["collection_names"].extend(
                [col_name]
                * len(col_results.get("ids", [[]])[0] if col_results.get("ids") else [])
            )

        except Exception as e:
            print(f"Error searching collection '{col_name}': {e}")
            continue  # Skip this collection and continue with others

    if not all_results["ids"]:
        print("No results found across any collections.")
        return all_results

    # Combine and sort results by distance (similarity) across all collections
    # Create a list of tuples (distance, metadata, document, id, collection_name) for sorting
    combined_results = list(
        zip(
            all_results["distances"],
            all_results["metadatas"],
            all_results["documents"],
            all_results["ids"],
            all_results["collection_names"],
        )
    )

    # Sort by distance (ascending, as lower distance means higher similarity)
    combined_results.sort(key=lambda x: x[0])

    # Extract top N results after sorting
    top_results = combined_results[:n_results]

    # Reconstruct the results dictionary with top N items
    final_results = {
        "metadatas": [item[1] for item in top_results],
        "documents": [item[2] for item in top_results],
        "distances": [item[0] for item in top_results],
        "ids": [item[3] for item in top_results],
        "collection_names": [
            item[4] for item in top_results
        ],  # Optional: include collection name in output
    }

    print(
        f"Found {len(final_results['ids'])} combined results from {len(set(final_results['collection_names']))} collection(s)."
    )
    return final_results


def get_all_job_ids(collection_name: str = "jobs"):
    """
    Get all stored job IDs from a specific collection.

    Args:
        collection_name: Name of the collection to query (default "remote_jobs")
    """
    client = chromadb.PersistentClient(path=CONFIG["chroma_db_path"])
    collection = client.get_collection(name=collection_name)
    results = collection.get(include=["ids"])
    return results["ids"]


def get_job_by_id(job_id: str, collection_name: str = "jobs"):
    """
    Retrieve a specific job by its ID from a specific collection.

    Args:
        job_id: Unique ID of the job to retrieve
        collection_name: Name of the collection to query (default "remote_jobs")
    """
    client = chromadb.PersistentClient(path=CONFIG["chroma_db_path"])
    collection = client.get_collection(name=collection_name)
    results = collection.get(ids=[job_id], include=["metadatas", "documents"])
    if results["metadatas"]:
        # Validate the retrieved metadata against the schema
        try:
            validated_metadata = JobMetadata(**results["metadatas"][0])
            return validated_metadata.model_dump()  # Return as dict for compatibility
        except Exception as e:
            print(
                f"Warning: Retrieved metadata for ID {job_id} does not match schema: {e}"
            )
            return results["metadatas"][0]  # Return raw metadata if validation fails
    return None


def add_job(job_dict: Dict[str, Any], collection_name: str = "jobs"):
    """
    Add a single job to a specific collection.

    Args:
        job_dict: Dictionary with job data (title, company, etc.)
        collection_name: Name of the collection to add the job to (default "remote_jobs")
    """
    client = chromadb.PersistentClient(path=CONFIG["chroma_db_path"])
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )
    model = SentenceTransformer(CONFIG["embedding_model"])  # Use model from config

    # Check for duplicate by URL
    existing = collection.get(where={"source_url": job_dict.get("link", "")})
    if existing["ids"]:
        print(
            f"Job with URL {job_dict.get('link', '')} already exists in collection '{collection_name}'."
        )
        return

    # Process job (reuse cleaning logic from main function)
    clean_desc = clean_description(job_dict.get("description", ""))
    company = job_dict.get("company", "") or extract_company_name(
        job_dict.get("description", "")
    )

    # Convert tags list to comma-separated string for ChromaDB metadata
    tags_value = job_dict.get("tags", [])
    if isinstance(tags_value, list):
        tags_str = ", ".join(str(tag) for tag in tags_value if tag is not None)
    else:
        tags_str = str(tags_value) if tags_value else ""

    # Prepare metadata - validate using Pydantic schema
    metadata_dict = {
        "job_id": str(uuid.uuid4()),
        "title": job_dict.get("title", ""),
        "company": company,
        "location": job_dict.get("region", ""),
        "description": clean_desc,
        "source_url": job_dict.get("link", ""),
        "posted_date": job_dict.get("published", ""),
        "tags": tags_str,  # Store as a string, not a list
        "ingested_at": datetime.utcnow().isoformat(),
    }

    # Validate metadata before adding
    try:
        validated_metadata = JobMetadata(**metadata_dict)
        final_metadata = validated_metadata.model_dump()
    except Exception as e:
        print(
            f"Error validating metadata for job '{job_dict.get('title', 'Unknown')}': {e}. Skipping."
        )
        return

    # Create embedding
    embedding_text = f"{job_dict.get('title', '')} — {company} — {clean_desc}"
    embedding = model.encode(embedding_text).tolist()

    # Add to collection
    collection.add(
        embeddings=[embedding],
        metadatas=[final_metadata],
        ids=[final_metadata["job_id"]],
    )
    print(f"Added job: {job_dict.get('title', '')} to collection '{collection_name}'")


def delete_job(job_id: str, collection_name: str = "jobs"):
    """
    Delete a job by its ID from a specific collection.

    Args:
        job_id: Unique ID of the job to delete
        collection_name: Name of the collection to delete from (default "remote_jobs")
    """
    client = chromadb.PersistentClient(path=CONFIG["chroma_db_path"])
    collection = client.get_collection(name=collection_name)

    collection.delete(ids=[job_id])
    print(f"Deleted job with ID: {job_id} from collection '{collection_name}'")


def delete_job_by_url(source_url: str, collection_name: str = "jobs"):
    """
    Delete a job by its source URL from a specific collection.

    Args:
        source_url: URL of the job listing to delete
        collection_name: Name of the collection to delete from (default "remote_jobs")
    """
    client = chromadb.PersistentClient(path=CONFIG["chroma_db_path"])
    collection = client.get_collection(name=collection_name)

    # Find job ID by URL
    results = collection.get(where={"source_url": source_url})
    if results["ids"]:
        collection.delete(ids=results["ids"])
        print(f"Deleted job with URL: {source_url} from collection '{collection_name}'")
    else:
        print(f"No job found with URL: {source_url} in collection '{collection_name}'")


def clear_job_database(collection_name: str = "jobs", db_path: str = None):
    """
    Delete all jobs from the specified collection in ChromaDB.

    Args:
        collection_name: Name of the collection to clear (default "remote_jobs")
        db_path: Path to the persistent ChromaDB directory (uses config if None)
    """
    if db_path is None:
        db_path = CONFIG["chroma_db_path"]

    client = chromadb.PersistentClient(path=db_path)

    # Use get_or_create_collection to ensure we're working with the right collection object
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine"
        },  # Use same metadata as ingestion if applicable
    )

    # Get all IDs in the collection
    all_results = collection.get(include=[])  # Fetch only IDs
    all_ids = all_results["ids"]

    if all_ids:
        # Delete all documents by their IDs
        collection.delete(ids=all_ids)
        print(f"Deleted {len(all_ids)} jobs from the collection '{collection_name}'.")
    else:
        print(f"Collection '{collection_name}' is already empty.")


def embed_resume_text(text: str) -> List[float]:
    """
    Embeds resume text using the same sentence transformer model
    and configuration as used for job ingestion.
    Ensures consistency between resume and job embeddings.
    """

    # Reuse model path/cache like in ingestor.py
    model = SentenceTransformer(CONFIG["embedding_model"])
    embedding = model.encode([text], normalize_embeddings=True)[0].tolist()
    return embedding


def query_jobs_by_embedding(
    embedding: List[float],
    n_results: int = 5,
    collection_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Query job collections using a precomputed embedding (e.g., from a resume).

    Args:
        embedding: Precomputed embedding vector (list of floats)
        n_results: Total number of top results to return across all collections
        collection_names: List of collection names to search; if None, searches all

    Returns:
        Dict with keys: "metadatas", "documents", "distances", "ids", "collection_names"
    """
    client = chromadb.PersistentClient(path=CONFIG["chroma_db_path"])

    if collection_names is None:
        all_collections = client.list_collections()
        collection_names = [col.name for col in all_collections]
        print(f"Querying all collections: {collection_names}")
    else:
        print(f"Querying specified collections: {collection_names}")

    if not collection_names:
        print("No collections found or specified.")
        return {
            "metadatas": [],
            "documents": [],
            "distances": [],
            "ids": [],
            "collection_names": [],
        }

    all_results = {
        "metadatas": [],
        "documents": [],
        "distances": [],
        "ids": [],
        "collection_names": [],
    }

    for col_name in collection_names:
        try:
            collection = client.get_collection(name=col_name)
            col_results = collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                include=["metadatas", "documents", "distances"],
            )
            # Safely extract results
            metadatas = (
                col_results.get("metadatas", [[]])[0]
                if col_results.get("metadatas")
                else []
            )
            documents = (
                col_results.get("documents", [[]])[0]
                if col_results.get("documents")
                else []
            )
            distances = (
                col_results.get("distances", [[]])[0]
                if col_results.get("distances")
                else []
            )
            ids = col_results.get("ids", [[]])[0] if col_results.get("ids") else []

            all_results["metadatas"].extend(metadatas)
            all_results["documents"].extend(documents)
            all_results["distances"].extend(distances)
            all_results["ids"].extend(ids)
            all_results["collection_names"].extend([col_name] * len(ids))

        except Exception as e:
            print(f"Error querying collection '{col_name}': {e}")
            continue

    if not all_results["ids"]:
        print("No results found for the given embedding.")
        return all_results

    # Combine and sort by distance (lower = more similar)
    combined = list(
        zip(
            all_results["distances"],
            all_results["metadatas"],
            all_results["documents"],
            all_results["ids"],
            all_results["collection_names"],
        )
    )
    combined.sort(key=lambda x: x[0])
    top = combined[:n_results]

    return {
        "metadatas": [t[1] for t in top],
        "documents": [t[2] for t in top],
        "distances": [t[0] for t in top],
        "ids": [t[3] for t in top],
        "collection_names": [t[4] for t in top],
    }


def example_usage():
    print("--- Example Usage ---")

    # Example 1: Search Jobs (across all collections)
    print("\n1. Searching for 'software engineer':")
    results = search_jobs("software engineer", n_results=3)
    if results["metadatas"]:
        for i, metadata in enumerate(results["metadatas"]):
            print(f"  Job {i+1}: {metadata['title']} at {metadata['company']}")
            print(f"    Location: {metadata['location']}")
            print(f"    URL: {metadata['source_url']}")
            print(
                f"    Collection: {results['collection_names'][i] if 'collection_names' in results else 'N/A'}"
            )
            print(f"    Distance: {results['distances'][i]:.4f}")
            print()
    else:
        print("  No results found.")

    # Example 2: Get all job IDs from the default collection
    print("\n2. Listing all job IDs in 'remote_jobs' collection:")
    all_ids = get_all_job_ids()
    print(f"  Found {len(all_ids)} job IDs: {all_ids[:5]}...")  # Print first 5 IDs

    # Example 3: Get a specific job by ID (replace with a real ID from your DB if available)
    print("\n3. Retrieving a specific job by ID:")
    # example_id = all_ids[0] if all_ids else "non_existent_id"
    # job_data = get_job_by_id(example_id)
    # if job_data:
    #     print(f"  Retrieved job: {job_data}")
    # else:
    #     print("  Job not found or ID list was empty.")
    print("  (Skipping - requires a real job ID. Get one from the list above.)")

    # Example 4: Add a new job (using example data)
    print("\n4. Adding a new example job:")
    example_job = {
        "title": "Example Python Developer",
        "region": "Remote",
        "published": "2025-10-18T15:00:00Z",
        "link": "https://example.com/job/999",
        "tags": ["Python", "FastAPI", "Docker"],
        "description": "An example job for testing purposes.",
        "company": "Example Corp",
    }
    add_job(example_job)  # Adds to 'remote_jobs' collection by default

    # Example 5: Delete the job we just added (by URL)
    print("\n5. Deleting the example job by URL:")
    delete_job_by_url("https://example.com/job/999")

    # Example 6: Clear the database (commented out for safety)
    # print("\n6. Clearing the 'remote_jobs' collection:")
    # clear_job_database() # Clears 'remote_jobs' collection by default
    print("\n6. Example for clearing database: clear_job_database() # Uncomment to run")
