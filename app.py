import os
from dotenv import load_dotenv
import streamlit as st
import fitz
from collections import Counter
import spacy
from spacy.cli import download as spacy_download
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize NLP model
@st.cache_resource
def load_nlp_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        spacy_download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = load_nlp_model()

# Function to extract text from PDF
def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text_parts = []
        for page in document:
            text_parts.append(page.get_text())
        pdf_text_content = " ".join(text_parts)
        return pdf_text_content
    else:
        raise FileNotFoundError("No file uploaded")

# Function to extract top keywords from the text
def extract_keywords(text):
    doc = nlp(text)
    job_related_stopwords = {'responsibilities', 'experience', 'requirements', 'qualifications'}
    keywords = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop and token.text.lower() not in job_related_stopwords]
    return Counter(keywords).most_common(10)  # Top 10 keywords

# Function to generate response from Gemini model
@st.cache_data
def get_gemini_response(input, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([input, pdf_content, prompt])
    return response.text

# Custom CSS for styling and responsiveness
st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .stButton>button { width: 100%; }
    .css-1d391kg { padding-top: 1rem; }
    @media (max-width: 768px) { .stApp { padding: 1rem; } }
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Single Resume Analysis", "Resume Comparison"])

if page == "Single Resume Analysis":
    st.title("JobFit Analyzer - Single Resume Analysis")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Job Description")
        input_text = st.text_area("Enter Job Description:", height=150)
        uploaded_jd_file = st.file_uploader("Or Upload Job Description (PDF)...", type=["pdf"], key="jd_upload")

        st.subheader("Resume")
        uploaded_file = st.file_uploader("Upload your Resume (PDF)...", type=["pdf"], key="resume_upload")

    with col2:
        st.subheader("Analysis Options")
        submit1 = st.button("📄 Tell Me About the Resume")
        submit2 = st.button("🛠️ How Can I Improve My Skills")
        submit3 = st.button("🔍 What Keywords Are Missing")
        submit4 = st.button("📊 Percentage Match")
        input_prompt = st.text_input("💬 Custom Query:")
        submit5 = st.button("🤔 Answer My Query")

    if uploaded_jd_file is not None:
        input_text = input_pdf_setup(uploaded_jd_file)

    if uploaded_file is not None:
        st.success("Resume PDF Uploaded Successfully")

    # Enhanced Prompts
    input_prompt1 = """
    As an experienced Human Resource Manager, your role is to assess the provided resume in relation to the job description.
    Evaluate the candidate's qualifications, experiences, and skills against the specified requirements. 
    Please highlight the strengths that align well with the role and any weaknesses or gaps that may need addressing.
    """

    input_prompt2 = """
    You are a Human Resource Manager with expertise in evaluating talent across various fields.
    Carefully analyze the resume in the context of the job description provided. 
    Share your insights regarding the candidate's fit for the role, and offer constructive feedback on areas for improvement and skill enhancement.
    """

    input_prompt3 = """
    You are an ATS (Applicant Tracking System) specialist. Evaluate the resume against the job description.
    Identify any critical keywords that are missing from the resume and suggest improvements to ensure the candidate's profile stands out.
    Provide additional recommendations for enhancing the candidate's overall presentation and alignment with the role.
    """

    input_prompt4 = """
    You are an ATS expert tasked with evaluating the compatibility of the resume with the provided job description.
    Calculate the percentage match between the two documents. 
    Additionally, list the missing keywords and provide final thoughts on the candidate's suitability for the role, including any suggestions for strengthening their application.
    """

    # Function to handle button actions
    def handle_button_action(button, prompt):
        if uploaded_file is not None and input_text:
            with st.spinner("Analyzing the resume..."):
                pdf_content = input_pdf_setup(uploaded_file)
                response = get_gemini_response(prompt, pdf_content, input_text)
                st.subheader("Analysis Result")
                st.write(response)

                # Download report
                report_content = f"**Resume Analysis Report**\n\n**Job Description:**\n{input_text}\n\n**Resume Content:**\n{pdf_content}\n\n**Analysis Result:**\n{response}"
                st.download_button(
                    label="Download Analysis Report",
                    data=report_content,
                    file_name="analysis_report.txt",
                    mime="text/plain"
                )
                
                if button == submit3:
                    keywords = extract_keywords(pdf_content)
                    st.subheader("Top Keywords")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=[count for _, count in keywords], y=[word for word, _ in keywords], ax=ax)
                    plt.title("Top Keywords in Resume")
                    plt.xlabel("Frequency")
                    st.pyplot(fig)
        else:
            st.warning("Please upload both the resume and job description to proceed.")

    # Button Actions
    if submit1:
        handle_button_action(submit1, input_prompt1)
    elif submit2:
        handle_button_action(submit2, input_prompt2)
    elif submit3:
        handle_button_action(submit3, input_prompt3)
    elif submit4:
        handle_button_action(submit4, input_prompt4)
    elif submit5:
        handle_button_action(submit5, input_prompt)

    st.markdown("""---""")

else:
    # Resume Comparison
    st.title("JobFit Analyzer - Resume Comparison")

    # Job Description Input
    st.subheader("Job Description")
    input_text = st.text_area("Enter Job Description (or upload as PDF):", height=150)
    uploaded_jd_file = st.file_uploader("Or Upload Job Description (PDF)...", type=["pdf"], key="jd_upload_compare")

    if uploaded_jd_file is not None:
        st.success("Job Description Uploaded Successfully")
        input_text = input_pdf_setup(uploaded_jd_file)

    # Select number of resumes
    n_resumes = st.number_input("Select the number of resumes to compare (2-10):", 2, 10, 2)

    # Upload multiple resumes
    uploaded_files = []
    for i in range(n_resumes):
        file = st.file_uploader(f"Upload Resume {i + 1} (PDF)", type=["pdf"], key=f"resume_{i}")
        if file is not None:
            uploaded_files.append(file)

    if st.button("Compare Resumes"):
        if input_text and len(uploaded_files) > 0:
            for resume_file in uploaded_files:
                pdf_content = input_pdf_setup(resume_file)
                response = get_gemini_response(input_prompt1, pdf_content, input_text)
                st.subheader(f"Analysis Result for Resume {uploaded_files.index(resume_file) + 1}")
                st.write(response)

                # Download report for each resume
                report_content = f"**Resume Comparison Report for Resume {uploaded_files.index(resume_file) + 1}**\n\n**Job Description:**\n{input_text}\n\n**Resume Content:**\n{pdf_content}\n\n**Analysis Result:**\n{response}"
                st.download_button(
                    label=f"Download Analysis Report for Resume {uploaded_files.index(resume_file) + 1}",
                    data=report_content,
                    file_name=f"comparison_report_resume_{uploaded_files.index(resume_file) + 1}.txt",
                    mime="text/plain"
                )
        else:
            st.warning("Please ensure a job description and at least one resume is uploaded for comparison.")

footer = """
---
#### Made By [Chaitanya](https://www.linkedin.com/in/naga-chaitanya-chowlur/)
For Queries, Reach out on [LinkedIn](https://www.linkedin.com/in/naga-chaitanya-chowlur/)  
*Resume Mastery: Your Gateway to Job Application Success*
"""

st.markdown(footer, unsafe_allow_html=True)
