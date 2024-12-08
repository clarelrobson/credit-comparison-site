import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

# Set Streamlit page layout to "wide"
st.set_page_config(layout="wide")

# Initialize the NLP model (paraphrase-MiniLM-L3-v2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=device)

# Compare the course description with courses from the selected university using Sentence Transformers
def compare_courses_batch(sending_course_desc, receiving_course_descs):
    # Encode the sending course description
    sending_course_vec = model.encode(sending_course_desc, convert_to_tensor=True, device=device)

    # Encode all receiving course descriptions in batch
    receiving_descriptions = list(receiving_course_descs.values())
    receiving_course_vecs = model.encode(receiving_descriptions, convert_to_tensor=True, device=device, batch_size=32)

    # Compute cosine similarities for all pairs at once
    similarity_scores = util.pytorch_cos_sim(sending_course_vec, receiving_course_vecs)

    # Create a dictionary of similarity scores
    results = {title: similarity_scores[0][i].item() for i, title in enumerate(receiving_course_descs.keys())}

    # Sort by similarity score (highest first) and return the top 10
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    return sorted_results[:10]

# Streamlit Interface
def main():
    # Create two full-width columns with a slight gap
    col1, col2 = st.columns([1.8, 1.2])  # Adjust the proportions to balance content width

    with col1:
        # Left Column: Title and Inputs
        st.title("Course Similarity Rater")
        st.markdown("""
        This site allows you to see how a course at one university (the sending university) might compare to courses from a different university (the receiving university). 
        It uses natural language processing (NLP) techniques to find the most similar courses based on their descriptions.
        """)

        # User input for the sending course description
        sending_course_desc = st.text_area("Enter the description for the sending university course")

        # Dropdown to select the university
        university = st.selectbox("Select the receiving university", ["Select...", "Pennsylvania State University", "Temple University", "West Chester University of PA"])

        # Similarity Rating Explanation (Default font size)
        st.markdown("""
        ## Similarity Rating Explanation
        The similarity rating is a value between 0 and 1 that indicates how closely the course description you provided matches each course in the database. 
        - **0.8 - 1.0**: Very High Similarity – The descriptions are nearly identical, with minimal difference.
        - **0.6 - 0.8**: High Similarity – The descriptions are very similar, but there may be some differences.
        - **0.4 - 0.6**: Moderate Similarity – The descriptions have noticeable differences, but share common topics or structure.
        - **0.2 - 0.4**: Low Similarity – The descriptions have some overlapping content, but are generally quite different.
        - **0.0 - 0.2**: Very Low Similarity – The descriptions are largely different with little to no overlap.
        """)

    with col2:
        # Right Column: Results
        if sending_course_desc and university != "Select...":
            # URLs for the university course CSV files
            psu_courses_file_url = "https://raw.githubusercontent.co
