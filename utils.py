import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
from bs4 import BeautifulSoup

def extract_resume_text(pdf_file):
    return extract_text(pdf_file)

def preprocess(text):
    return text.lower().replace('\n', ' ').strip()

def fetch_remoteok_jobs():
    url = "https://remoteok.com/api"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    data = response.json()

    jobs = []
    for item in data:
        if isinstance(item, dict) and item.get("position") and item.get("description") and item.get("url"):
            job_url = item.get("url", "")
            if not job_url.startswith("https://"):
                job_url = "https://remoteok.com" + job_url

            jobs.append({
                "title": item.get("position", ""),
                "company": item.get("company", ""),
                "description": item.get("description", ""),
                "url": job_url
            })

    return pd.DataFrame(jobs)

def fetch_microsoft_jobs():
    url = "https://careers.microsoft.com/us/en/search-results"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    jobs = []
    for job_card in soup.find_all('section', class_="jobs-list-container"):
        title_tag = job_card.find('h3')
        if title_tag:
            title = title_tag.text.strip()
            link_tag = job_card.find('a')
            if link_tag:
                link = link_tag['href']
                full_link = "https://careers.microsoft.com" + link
                jobs.append({
                    "title": title,
                    "company": "Microsoft",
                    "description": "Details on Microsoft Career Page",
                    "url": full_link
                })
    return pd.DataFrame(jobs)

def fetch_angellist_jobs():
    url = "https://wellfound.com/jobs"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    jobs = []
    job_cards = soup.find_all('div', class_='styles_component__P6AY4')  # New AngelList CSS classes
    for card in job_cards:
        title_tag = card.find('h2')
        company_tag = card.find('h3')
        link_tag = card.find('a', href=True)

        if title_tag and link_tag:
            title = title_tag.text.strip()
            company = company_tag.text.strip() if company_tag else "Startup"
            link = link_tag['href']
            if not link.startswith('https://'):
                link = "https://wellfound.com" + link

            jobs.append({
                "title": title,
                "company": company,
                "description": "Details on AngelList Job Page",
                "url": link
            })

    return pd.DataFrame(jobs)

def fetch_all_jobs():
    remoteok = fetch_remoteok_jobs()

    try:
        microsoft = fetch_microsoft_jobs()
    except Exception as e:
        print(f"Error fetching Microsoft jobs: {e}")
        microsoft = pd.DataFrame(columns=["title", "company", "description", "url"])

    try:
        angellist = fetch_angellist_jobs()
    except Exception as e:
        print(f"Error fetching AngelList jobs: {e}")
        angellist = pd.DataFrame(columns=["title", "company", "description", "url"])

    all_jobs = pd.concat([remoteok, microsoft, angellist], ignore_index=True)
    return all_jobs

def match_resume_to_jobs(resume_text, jobs_df):
    if jobs_df.empty:
        return pd.DataFrame(columns=['title', 'company', 'similarity', 'url'])

    resume_clean = preprocess(resume_text)
    jobs_df['processed'] = jobs_df['description'].apply(preprocess)

    corpus = [resume_clean] + jobs_df['processed'].tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    jobs_df['similarity'] = similarities

    top_matches = jobs_df.sort_values(by='similarity', ascending=False).head(10)
    return top_matches[['title', 'company', 'similarity', 'url']]
