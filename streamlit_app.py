import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

# Initialize the NLP model (paraphrase-MiniLM-L3-v2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=device)

# Compare the course description with PSU courses using Sentence Transformers (optimized)
def compare_courses_batch(sending_course_desc, receiving_course_descs):
    # Encode the sending course description
    sending_course_vec = model.encode(sending_course_desc, convert_to_tensor=True, device=device)

    # Encode all receiving course descriptions in batch
    receiving_courses_descs = list(receiving_course_descs.values())
    receiving_course_vecs = model.encode(receiving_courses_descs, convert_to_tensor=True, device=device, batch_size=32)

    # Compute cosine similarities for all pairs at once
    similarity_scores = util.pytorch_cos_sim(sending_course_vec, receiving_course_vecs)

    # Create a dictionary of similarity scores
    results = {title: similarity_scores[0][i].item() for i, title in enumerate(receiving_course_descs.keys())}

    # Sort by similarity score (highest first) and return the top 10
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    return sorted_results[:10]

# Streamlit Interface
def main():
    st.title("Course Description Similarity Finder")

    # User input for the sending course description
    sending_course_desc = st.text_area("Enter the course description you want to compare:")

    # Dropdown to select the university
    university = st.selectbox("Select University", ["Penn State"])

    # Check if the user selected a university and provided a course description
    if sending_course_desc and university:
        # Load the corresponding CSV file for the selected university
        if university == "Penn State":
            psu_courses_file_path = '/content/drive/MyDrive/PSU/fall24/ds440/psu_courses.csv'  # Update with your file path
            psu_courses_df = pd.read_csv(psu_courses_file_path)

            if 'Course Title' not in psu_courses_df.columns or 'Description' not in psu_courses_df.columns:
                st.error("Penn State courses CSV must contain 'Course Title' and 'Description' columns.")
                return
            
            # Prepare a dictionary of Penn State course titles and descriptions
            psu_courses = dict(zip(psu_courses_df['Course Title'], psu_courses_df['Description']))

            # Compare the sending course description with Penn State courses
            top_10_courses = compare_courses_batch(sending_course_desc, psu_courses)

            # Display the results
            st.subheader("Top 10 Most Similar Penn State Courses:")
            for course_title, score in top_10_courses:
                st.write(f"Course: {course_title}, Similarity Score: {score}")
    else:
        st.warning("Please enter a course description and select a university.")

# Run the Streamlit app
if __name__ == "__main__":
    main()

