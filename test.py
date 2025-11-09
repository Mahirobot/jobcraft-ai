import scraper
import ingestor
job_list, job_collection = scraper.remotive_scraper()
ingestor.ingest_jobs_to_rag(job_list, job_collection)
job_list, job_collection = scraper.we_work_remotely_scraper()
ingestor.ingest_jobs_to_rag(job_list, job_collection)
job_list, job_collection = scraper.real_work_from_anywhere_scraper()
ingestor.ingest_jobs_to_rag(job_list, job_collection)
job_list, job_collection = scraper.empllo_jobs_scraper()
ingestor.ingest_jobs_to_rag(job_list, job_collection)