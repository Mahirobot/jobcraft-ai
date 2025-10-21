import feedparser
from bs4 import BeautifulSoup
from html import unescape
import schemas  # Assuming your JobEntry model is defined here
import re
from config import CONFIG

def clean_description(desc: str) -> str:
    """
    Remove HTML tags, extra whitespace, and common boilerplate from job descriptions.
    """
    # Remove HTML tags
    soup = BeautifulSoup(desc, "html.parser")
    clean_text = soup.get_text(separator=" ")
    
    # Remove common boilerplate patterns
    patterns_to_remove = [
        r'URL:\s*https?://[^\s]+',
        r'Headquarters:\s*[^\n]+',
        r'Company:\s*[^\n]+',
        r'Website:\s*https?://[^\s]+',
        r'Visit our website at https?://[^\s]+',
        r'Apply at https?://[^\s]+',
        r'For more information: https?://[^\s]+',
        r'Apply Now?://[^\s]+',
    ]
    
    for pattern in patterns_to_remove:
        clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text.strip()


def extract_company_name(desc: str) -> str:
    """
    Extract company name from description if present in common formats.
    """
    url_match = re.search(r'URL:\s*(https?://[^\s]+)', desc, re.IGNORECASE)
    if url_match:
        url = url_match.group(1)
        import urllib.parse
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc
        parts = domain.split('.')
        if len(parts) >= 2:
            return parts[0]
    
    # Look for "Company:" or "Headquarters:" patterns
    company_match = re.search(r'(?:Company|Headquarters):\s*([^\n]+)', desc, re.IGNORECASE)
    if company_match:
        return company_match.group(1).strip()
    
    return "Unknown"


def we_work_remotely_scraper():
    rss_url = CONFIG["rss_feeds"]["we_work_remotely"]["link"]
    feed = feedparser.parse(rss_url)
    job_list = []
    
    for entry in feed.entries:
        # Extract and clean description
        soup = BeautifulSoup(unescape(entry.description), "html.parser")
        description = soup.get_text(strip=False)
        clean_desc = clean_description(description)
        
        raw_tags = getattr(entry, 'tags', [])
        processed_tags = [tag_dict.get('term', '').strip() for tag_dict in raw_tags if tag_dict.get('term')]
        processed_tags = ", ".join(processed_tags)

        company = entry.get('company', '')
        if not company or company == "Unknown":
            company = extract_company_name(description)

        job_data = {
            "title": entry.get("title", "").strip(),
            "link": entry.get("link", "").strip(),
            "published": entry.get("published", ""), 
            "region": getattr(entry, "region", "Not Specified").strip(),
            "tags": processed_tags,
            "description": clean_desc,
            "company": company,
        }
        
        # Validate and create the job entry
        validated_job = schemas.JobEntry(**job_data)
        job_list.append(validated_job.model_dump())
    return job_list, CONFIG["rss_feeds"]["we_work_remotely"]["collection_name"]


def remotive_scraper():
    rss_url = CONFIG["rss_feeds"]["remotive"]["link"]
    feed = feedparser.parse(rss_url)
    job_list = []
    
    for entry in feed.entries:
        # Extract and clean description
        soup = BeautifulSoup(unescape(entry.description), "html.parser")
        description = soup.get_text(strip=False)
        clean_desc = clean_description(description)
        
        raw_tags = getattr(entry, 'tags', [])
        processed_tags = [tag_dict.get('term', '').strip() for tag_dict in raw_tags if tag_dict.get('term')]
        processed_tags = ", ".join(processed_tags)

        company = entry.get('company', '')
        if not company or company == "Unknown":
            company = extract_company_name(description)

        job_data = {
            "title": entry.get("title", "").strip(),
            "link": entry.get("link", "").strip(),
            "published": entry.get("published", ""), 
            "region": getattr(entry, "location", "Not Specified").strip(),
            "tags": processed_tags,
            "description": clean_desc,
            "company": company,
        }
        
        # Validate and create the job entry
        validated_job = schemas.JobEntry(**job_data)
        job_list.append(validated_job.model_dump())
    return job_list, CONFIG["rss_feeds"]["remotive"]["collection_name"]
