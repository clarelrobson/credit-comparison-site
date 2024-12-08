import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Initialize model
model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

# Sample data (you would load your CSV file here)
df = pd.read_csv('/content/drive/MyDrive/PSU/fall24/ds440/psu_courses.csv')

# Compare function
def compare_courses(sending_course_desc, receiving_course_descs):
    sending_course_vec = model.encode(sending_course_desc, convert_to_tensor=True)
    receiving_courses_descs = list(receiving_course_descs.values())
    receiving_course_vecs = model.encode(receiving_courses_descs, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(sending_course_vec, receiving_course_vecs)
    results = {title: similarity_scores[i].item() for i, title in enumerate(receiving_course_descs.keys())}
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    return sorted_results[:10]

# Streamlit interface
st.title("Course Similarity Comparison")
st.write("Enter the course description and select a university to find the most similar courses.")

# Input fields
course_desc = st.text_area("Enter Course Description:")
university = st.selectbox("Select University", ['PennState', 'Other'])

if st.button("Compare"):
    if course_desc:
        # For simplicity, assume you have a method to get courses based on university
        receiving_courses = {row['Course Title']: row['Course Description'] for index, row in df.iterrows()}
        top_10_courses = compare_courses(course_desc, receiving_courses)

        st.write("Top 10 most similar courses:")
        for course, score in top_10_courses:
            st.write(f"{course}: Similarity Score: {score}")
    else:
        st.error("Please enter a course description.")
