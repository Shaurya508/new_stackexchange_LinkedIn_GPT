import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
def find_most_relevant_image(excel_path, user_question):
    """
    Find the most relevant image from an Excel file based on a user's question.
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_path)
        print(f"DataFrame loaded. Shape: {df.shape}")
        print("Columns:", df.columns.tolist())

        # Validate required columns
        if 'Slide number' not in df.columns or 'Topic' not in df.columns:
            raise ValueError("Excel file must contain 'Slide number' and 'Topic' columns")

        # Preprocess function
        def preprocess_text(text):
            if pd.isna(text):
                return ""
            # Convert to lowercase and remove special characters
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            return text
        
        # Preprocess user question and topics
        processed_question = preprocess_text(user_question)
        print(f"Processed question: {processed_question}")

        df['processed_topic'] = df['Topic'].apply(preprocess_text)
        print("Processed topics sample:", df['processed_topic'].head())

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        
        # Combine processed question with processed topics
        all_texts = [processed_question] + df['processed_topic'].tolist()
        
        # Compute TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Compute cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        
        # Find the index of the most similar topic
        most_similar_index = cosine_similarities.argmax()
        
        # Print similarity scores for debugging
        print("Similarity scores:", cosine_similarities)
        print(f"Most similar index: {most_similar_index}")
        
        # Return the corresponding slide number (image name)
        return df.iloc[most_similar_index]['Slide number']

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    user_question = "MMM Contribution Chart"
    excel_path = 'Image_clarifications.xlsx'
    most_relevant_image = find_most_relevant_image(excel_path,user_question)
    print(f"The most relevant image for the question '{user_question}' is: {most_relevant_image}")

main()