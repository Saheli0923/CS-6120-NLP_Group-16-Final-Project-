from __future__ import annotations

import streamlit as st
from PyPDF2 import PdfReader
from predict_score import predict_rating
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Downloading the NLTK stopwords to use for filtering out stopwords
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lower case
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    return tokens



# ---------------------------------
st.set_page_config(page_title='📝 Resume Ratings', page_icon="📝")
st.title("📝 Resume Ratings")
st.markdown("Use this application to help you decide if the prospect is a good fit for the job.")

with st.form(key='resume_form'):
    job_description = st.text_area(label="""Write the Job Description here.
                                            Insert key aspects you want to value in the prospect's resume.""",
                                   placeholder="Job description. This field should have at least 100 characters.")
    file = st.file_uploader("Add the prospect's resume in PDF format:", type=["pdf"])
   
    submitted = st.form_submit_button('Submit')

if file is not None and len(job_description) > 100:
    pdf_file = PdfReader(file)
    pdf_text = ""
    for page in pdf_file.pages:
        pdf_text += page.extract_text() + "\n"


    preprocessed_resume = ''.join(preprocess_text(pdf_text[750:]))
    

    # Get the resume rating using the fine-tuned model
    resume_rating = predict_rating(job_description, preprocessed_resume)

    st.title("Resume Score with Finetuned Model")
    st.markdown(f"#### Resume Rating: {round(resume_rating,2)}")