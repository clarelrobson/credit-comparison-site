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
    # Custom CSS to change the fonts
    st.markdown("""
    <style>
    /* Load Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&family=Pacifico&display=swap');

    /* Apply fun fonts to the header and body */
    h1, h2, h3 {
        font-family: 'Pacifico', cursive;
    }
    body {
        font-family: 'Poppins', sans-serif;
    }
    .stTextArea textarea {
        font-family: 'Poppins', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

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

        # Similarity Rating Explanation (with a red box around it)
        st.markdown("""
        <div style="border: 2px solid red; background-color: #f8d7da; padding: 10px;">
            <h3>Similarity Rating Explanation:</h3>
            <ul>
                <li><strong>0.8 - 1.0</strong>: Very High Similarity – The descriptions are nearly identical, with minimal difference.</li>
                <li><strong>0.6 - 0.8</strong>: High Similarity – The descriptions are very similar, but there may be some differences.</li>
                <li><strong>0.4 - 0.6</strong>: Moderate Similarity – The descriptions have noticeable differences, but share common topics or structure.</li>
                <li><strong>0.2 - 0.4</strong>: Low Similarity – The descriptions have some overlapping content, but are generally quite different.</li>
                <li><strong>0.0 - 0.2</strong>: Very Low Similarity – The descriptions are largely different with little to no overlap.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Right Column: Results
        if sending_course_desc and university != "Select...":
            # URLs for the university course CSV files
            psu_courses_file_url = "https://raw.githubusercontent.com/clarelrobson/credit-comparison-site/main/psu_courses_with_credits.csv"
            temple_courses_file_url = "https://raw.githubusercontent.com/clarelrobson/credit-comparison-site/main/temple_courses_with_credits.csv"
            wcu_courses_file_url = "https://raw.githubusercontent.com/clarelrobson/credit-comparison-site/main/wcu_courses.csv"

            # Load the selected university's course descriptions CSV
            try:
                if university == "Pennsylvania State University":
                    courses_file_url = psu_courses_file_url
                elif university == "Temple University":
                    courses_file_url = temple_courses_file_url
                elif university == "West Chester University of PA":
                    courses_file_url = wcu_courses_file_url

                courses_df = pd.read_csv(courses_file_url)

                # Check if the necessary columns are present
                required_columns = ['Course Title', 'Description']
                if not all(col in courses_df.columns for col in required_columns):
                    st.error(f"{university} courses CSV must contain the columns: {', '.join(required_columns)}.")
                    return

                # Prepare dictionaries for course titles and descriptions
                courses = dict(zip(courses_df['Course Title'], courses_df['Description']))

                # Compare the sending course description with the selected university's courses
                top_10_courses = compare_courses_batch(sending_course_desc, courses)

                # Display the results with the header
                st.subheader(f"Top 10 Most Similar {university} Courses")

                for course_title, score in top_10_courses:
                    st.write(f"**{course_title}** (Similarity Score: {score:.2f})")

            except Exception as e:
                st.error(f"Error loading courses: {e}")
        else:
            st.warning("Please enter a course description and select a university.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
