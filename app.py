# Importing libraries
import pandas as pd
import numpy as np
import re
import time
import os
import logging
from tqdm import tqdm
from textblob import TextBlob  # type: ignore
from transformers import AutoTokenizer, AutoModel  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from scipy.spatial.distance import cosine
from numpy.linalg import norm
import requests
import streamlit as st
import torch
from datetime import datetime
import pickle

# Set up logging
LOG_FILE = "app.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("Application started.")

# Load pre-trained model
logging.info("Loading Sentence Transformer model.")
device = torch.device("cpu")
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
model.to(device)

# Load pre-trained embeddings
logging.info("Loading text embeddings.")
# EMBEDDINGS_FILE = 'C:/Users/dishi/OneDrive/Desktop/DS-MBAN/ds2ass2/text_embeddings.pkl' #"text_embeddings.pkl"
# embeddings_dataset = pickle.load(open(EMBEDDINGS_FILE, 'rb'))
with open("export.pkl", "rb") as f:
    embeddings_dataset = pickle.load(f)

# API configurations
API_ENDPOINT = 'https://nubela.co/proxycurl/api/v2/linkedin'
API_KEY = 'XsOM3OP9dgTtWgA-_4eFnA'
HEADERS = {'Authorization': 'Bearer ' + API_KEY}

# For getting the vector embeddings
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

# Define helper functions
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    logging.info("Generating embeddings for the input text.")
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


def recommend_jobs_from_linkedin(embeddings_dataset, model):
    """
    Streamlit application for recommending tasks based on semantic similarity.
    """
    st.title("🔍 CareerMatchX ")

    linkedin_profile_url = st.text_input("🔗 Enter your LinkedIn profile URL:")

    if st.button("🚀 Find"):
        logging.info("Button clicked. Processing started.")
        st.write("⚙️ Processing your LinkedIn profile...")

        try:
            response = requests.get(
                API_ENDPOINT,
                params={'url': linkedin_profile_url, 'skills': 'include'},
                headers=HEADERS,
            )
            response.raise_for_status()
            profile_data = response.json()
            logging.info("LinkedIn profile data retrieved successfully.")
        except requests.RequestException as e:
            logging.error(f"Error retrieving LinkedIn profile: {e}")
            st.error("❌ Failed to retrieve LinkedIn profile. Please check the URL or try again later.")
            return

        # Process education
        education_data = profile_data['education']
        # Get today's date and calculate the cutoff year
        today = datetime.today()
        cutoff_year = today.year - 5

        # Filter data based on the year, handling missing 'ends_at'
        filtered_data = [
            {
                'degree_name': record['degree_name'],
                'school': record['school'],
                'description': record['description'],
                'grade': record['grade']
            }
            for record in education_data
            if (
                # Check if 'ends_at' exists and compare the year
                record.get('ends_at') and record['ends_at']['year'] >= cutoff_year
            ) or (
                # Include records where 'ends_at' is None (e.g., ongoing studies)
                record['starts_at']['year'] >= cutoff_year
            )
        ]
        # Limit to only 2 education records
        limited_education = filtered_data[:1]

        # Process experience
        today = datetime.today()
        cutoff_year = today.year - 2

        experience_data = profile_data['experiences']

        # Get today's date and calculate the cutoff year
        today = datetime.today()
        cutoff_year = today.year - 2

        # Filter for experiences within the last two years
        recent_experiences = [
            {
                'company': exp['company'],
                'title': exp['title'],
                'description': exp.get('description'),
                'location': exp.get('location'),
            }
            for exp in experience_data
            if exp['starts_at']['year'] >= cutoff_year or (
                exp['ends_at'] and exp['ends_at']['year'] >= cutoff_year
            )
        ]

        # Consolidate profile data
        consolidated_data = {
            "Summary": profile_data.get("summary", ""),
            "Headline": profile_data.get("headline", ""),
            "Country": profile_data.get("country", ""),
            "State": profile_data.get("state", ""),
            "City": profile_data.get("city", ""),
            "Certifications": ", ".join(cert.get("name", "") for cert in profile_data.get("certifications", [])),
            "Skills": ", ".join(profile_data.get("skills", [])),
            "Education": ", ".join(f"{edu['degree_name']} from {edu['school']}" for edu in limited_education),
            "Experience": ", ".join(f"{exp['title']} at {exp['company']}" for exp in recent_experiences),
        }
        logging.info("Profile data consolidated successfully.")
        
        # Generate embeddings
        consolidated_data = pd.DataFrame({key: [value] for key, value in consolidated_data.items()})
        consolidated_data['profile_linkedin_summary'] = consolidated_data['Summary'] + consolidated_data['Headline'] + consolidated_data['Country'] + consolidated_data['State'] +  consolidated_data['City'] +  consolidated_data['Certifications'] + consolidated_data['Skills'] + consolidated_data['Education'] + consolidated_data['Experience']
        ques = consolidated_data['profile_linkedin_summary'].tolist()
        profile_embedding = get_embeddings(ques).cpu().detach().numpy()
        # Find nearest examples
        scores, samples = embeddings_dataset.get_nearest_examples("embeddings", profile_embedding, k=5)
        samples_df = pd.DataFrame.from_dict(samples)
        samples_df["Score"] = scores

        # Compute cosine similarity
        samples_df["Matching Score"] = [
            np.dot(profile_embedding, embedding) / (norm(profile_embedding) * norm(embedding))
            for embedding in samples_df["embeddings"]
        ]
        samples_df["Matching Score"] = (samples_df["Matching Score"] * 100).round(2)
        samples_df["Matching Score"] = samples_df["Matching Score"].round(2)
        samples_df.sort_values("Matching Score", ascending=False, inplace=True)

        # Display top results
        st.subheader("🎯 Top Recommendations")
        print(samples_df.columns)
        samples_df = samples_df[['job_id', 'company_name', 'title', 'remote_allowed',
       'formatted_work_type', 'Country', 'City', 'State', 'Matching Score']]
        top_recommendations = samples_df.head(5)
        st.dataframe(top_recommendations)

# Run the application
if __name__ == "__main__":
    logging.info("Starting Streamlit application.")
    recommend_jobs_from_linkedin(embeddings_dataset, model)
