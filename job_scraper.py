import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, Table, Column, Integer, String, Text, MetaData

# Database setup
DB_PATH = "jobs.db"
engine = create_engine(f"sqlite:///{DB_PATH}")
metadata = MetaData()

# Define or create table
jobs_table = Table(
    'jobs', metadata,
    Column('id', Integer, primary_key=True),
    Column('title', String),
    Column('company', String),
    Column('description', Text)
)
metadata.create_all(engine)

def scrape_jobs_indeed(query="software engineer", location="remote", max_jobs=10):
    url = f"https://www.indeed.com/jobs?q={query}&l={location}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    job_cards = soup.find_all('a', class_='tapItem', limit=max_jobs)
    jobs = []

    for job in job_cards:
        try:
            title = job.find('h2', class_='jobTitle').get_text(strip=True)
            company = job.find('span', class_='companyName').get_text(strip=True)
            description = job.find('div', class_='job-snippet').get_text(strip=True)
            jobs.append({"title": title, "company": company, "description": description})
        except AttributeError:
            continue

    return jobs

def insert_jobs(jobs):
    with engine.connect() as conn:
        for job in jobs:
            conn.execute(jobs_table.insert().values(**job))
    print(f"{len(jobs)} jobs inserted.")

if __name__ == "__main__":
    jobs = scrape_jobs_indeed()
    insert_jobs(jobs)
