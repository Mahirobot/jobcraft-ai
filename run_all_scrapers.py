import ingestor as ingest
import scraper as sc
import logging
import sys
import os


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), "scraper.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_all_scrapers_24():
    collection_name = "jobs"
    ##################
    # Clear Database
    ##################

    ingest.clear_job_database(collection_name)

    ########################
    # Calling All Scrappers
    ########################
    all_job_list = []

    # calling scraper
    job_list1, _ = sc.real_work_from_anywhere_scraper()
    job_list3, _ = sc.we_work_remotely_scraper()
    job_list2, _ = sc.remotive_scraper()
    job_list4, _ = sc.empllo_jobs_scraper()

    # append to list
    all_job_list.append(job_list1)
    all_job_list.append(job_list2)
    all_job_list.append(job_list3)
    all_job_list.append(job_list4)

    # Adding to chromaDB
    logger.info("Adding Data to DB...")
    for jobs in all_job_list:
        ingest.ingest_jobs_to_rag(jobs, collection_name)

# Add the project directory to Python path (optional but safe)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":

    logger.info("Starting daily job scraping...")

    run_all_scrapers_24()

    logger.info("Scraping completed.")