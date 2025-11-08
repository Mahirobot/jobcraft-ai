from datetime import datetime
from typing import List, Literal, Optional

from dateutil import parser  # Import dateutil.parser
from pydantic import BaseModel, field_validator

COLLECTION_NAMES = Literal["jobs"]


class JobEntry(BaseModel):
    title: str
    link: str
    region: str
    published: str
    tags: Optional[str]
    description: str
    company: Optional[str] = None

    @field_validator("published")
    def validate_and_convert_date(cls, v):
        """
        Validates and converts the date string to ISO 8601 format.
        Handles various input formats using dateutil.parser.
        """
        if not v:
            return v

        try:
            parsed_date = parser.parse(v)
            return parsed_date.isoformat() + "Z"
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not parse date string '{v}': {e}")

    @field_validator("link")
    def validate_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("Link must be a valid URL")
        return v


class JobMetadata(BaseModel):
    """
    Schema for job metadata stored in ChromaDB.
    """

    job_id: str  # UUID as string
    title: str
    company: str
    location: str
    description: str
    source_url: str
    posted_date: str  # ISO format datetime
    tags: str  # Comma-separated string representation of tags list
    ingested_at: str
