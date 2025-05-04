import streamlit as st
from utils import extract_resume_text, fetch_all_jobs, match_resume_to_jobs
import pandas as pd

st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("📄 AI-Powered Resume Analyzer & Multi-Source Job Matcher")

st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #121212;
        color: #E0E0E0;
    }

    .header {
        font-size: 36px;
        color: #E0E0E0;
        margin-bottom: 20px;
    }

    .job-card {
        background-color: #1E1E1E;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .job-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }

    .job-card h3 {
        font-size: 22px;
        color: #E0E0E0;
        margin-bottom: 10px;
    }

    .job-card p {
        color: #B0B0B0;
        font-size: 16px;
        margin-bottom: 15px;
    }

    .apply-button {
        display: inline-block;
        padding: 12px 24px;
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        font-size: 16px;
        border-radius: 8px;
        text-decoration: none;
        text-align: center;
        transition: background-color 0.3s ease;
    }

    .apply-button:hover {
        background-color: #004d99;
    }

    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }

    .resume-section {
        margin-bottom: 30px;
    }

    .resume-section h2 {
        font-size: 24px;
        color: #E0E0E0;
        margin-bottom: 10px;
    }

    .text-area {
        background-color: #2C2C2C;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        color: #E0E0E0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header("Job Filters")
remote = st.sidebar.checkbox("Only Remote Jobs")
microsoft_jobs = st.sidebar.checkbox("Only Microsoft Jobs")
startups = st.sidebar.checkbox("Only Startups (AngelList)")
tech_stack = st.sidebar.text_input("Filter by Tech Stack (e.g., Python, JavaScript)")

# File uploader
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from resume..."):
        resume_text = extract_resume_text(uploaded_file)
        st.success("Resume text extracted!")

    st.subheader("Extracted Resume Content:")
    st.markdown('<div class="resume-section"><div class="text-area">', unsafe_allow_html=True)
    st.text_area("Resume Text", resume_text, height=250)
    st.markdown('</div></div>', unsafe_allow_html=True)

    with st.spinner("Fetching live job listings from multiple sources..."):
        jobs_df = fetch_all_jobs()

    # Make sure 'source' column exists
    if 'source' not in jobs_df.columns:
        jobs_df['source'] = jobs_df.apply(
            lambda row: 'AngelList' if 'angel' in row['company'].lower() else 'LinkedIn', axis=1
        )

    # Apply filters BEFORE matching
    if not jobs_df.empty:
        if remote:
            if 'remote' in jobs_df.columns:
                jobs_df = jobs_df[jobs_df['remote'] == True]
        if microsoft_jobs:
            jobs_df = jobs_df[jobs_df['company'].str.contains('Microsoft', case=False, na=False)]
        if startups:
            jobs_df = jobs_df[jobs_df['source'] == 'AngelList']
        if tech_stack:
            if 'tech_stack' in jobs_df.columns:
                jobs_df = jobs_df[jobs_df['tech_stack'].str.contains(tech_stack, case=False, na=False)]

        with st.spinner("Matching your resume with available jobs..."):
            matched_jobs = match_resume_to_jobs(resume_text, jobs_df)

        if not matched_jobs.empty:
            st.subheader("💼 Top Job Matches for You:")
            for _, row in matched_jobs.iterrows():
                st.markdown(f"""
                    <div class="job-card">
                        <h3>{row['title']} at {row['company']}</h3>
                        <p><strong>Similarity Score:</strong> {row['similarity']:.2f}</p>
                        <a class="apply-button" href="{row['url']}" target="_blank">🔗 Apply Now</a>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No jobs matched your resume after applying filters.")
    else:
        st.warning("⚠️ Could not fetch jobs. Try again later.")

else:
    st.info("Upload a PDF resume to get started.")
