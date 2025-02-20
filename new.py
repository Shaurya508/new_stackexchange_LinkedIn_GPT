from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
# from PyPDF2 import PdfReader #used it before now using tesseract
import requests
from bs4 import BeautifulSoup
# from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import pandas as pd
from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chains import ConversationChain
from pdf2image import convert_from_path
# from PIL import Image
import streamlit as st
import pytesseract
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import re
import langid
from trial import translate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    
    # Rejoin tokens
    processed_text = ' '.join(tokens)
    
    # Remove extra whitespaces
    processed_text = ' '.join(processed_text.split())
    
    return processed_text


# from langchain_ollama import ChatOllama, OllamaEmbeddings

# Load the Google API key from the .env file
# load_dotenv()
# API_KEY = "AIzaSyCvw_aGHyJtLxpZ4Ojy8EyaEDtPOzZM29"

# Configure Google Generative AI API key
# genai.configure(api_key=API_KEY)

# genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
# Load the Google API key from the .env file
# load_dotenv()
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
# genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# sec_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# os.environ["HUGGINGFACEHUB_API_TOKEN"]=sec_key
# Function to log in to LinkedIn


def find_most_relevant_image(user_question, excel_path='Image_clarifications.xlsx', threshold=0.2):
    """
    Find the most relevant image from an Excel file based on a user's question.

    Parameters:
    user_question (str): User's input question
    excel_path (str): Path to the Excel file
    threshold (float): Minimum similarity score required to return an image

    Returns:
    str or None: Name of the most relevant image or None if similarity is below the threshold
    """
    # Ensure stopwords are available
    nltk.download('stopwords', quiet=True)
    
    # Read the Excel file
    df = pd.read_excel(excel_path)

    # Check if 'Topic' and 'Slide number' columns exist
    if 'Topic' not in df.columns or 'Slide number' not in df.columns:
        raise ValueError("Excel file must contain 'Topic' and 'Slide number' columns")

    # Preprocess text function
    def preprocess_text(text):
        if pd.isna(text):
            return ""  # Handle NaN values
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)

    # Preprocess user question
    processed_question = preprocess_text(user_question)

    # Preprocess topics
    df['processed_topic'] = df['Topic'].apply(preprocess_text)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Combine processed question with processed topics
    all_texts = [processed_question] + df['processed_topic'].tolist()
    
    # Compute TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    # Find the highest similarity score
    max_similarity = cosine_similarities.max()
    print(max_similarity)
    # If similarity is below the threshold, return None
    if max_similarity < threshold:
        return None

    # Find the index of the most similar topic
    most_similar_index = cosine_similarities.argmax()

    # Return the corresponding slide number (image name)
    return df.iloc[most_similar_index]['Slide number']

def linkedin_login(email, password , driver):
    driver.get("https://www.linkedin.com/login")
    
    # Find the username/email field and send the email
    email_field = driver.find_element(By.ID, "username")
    email_field.send_keys(email)
    
    # Find the password field and send the password
    password_field = driver.find_element(By.ID, "password")
    password_field.send_keys(password)
    
    # Submit the form
    password_field.send_keys(Keys.RETURN)
    
    # Wait for a bit to allow login to complete
    time.sleep(5)



# def scrape_linkedin_post(url, driver):
#     # Open the LinkedIn post URL
#     driver.get(url)
#     try:
#         img_elements = driver.find_elements(By.TAG_NAME, 'img')

#     # Extract and print image URLs
#         image_url = [img.get_attribute('src') for img in img_elements]
#         # image_url = image_element.get_attribute('src')
#     except:
#         image_url = None
#     # Wait for the content to load
#     try:
#         post_content = WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.CLASS_NAME, 'feed-shared-update-v2__description'))
#         )
#         return post_content.text.encode('ascii', 'ignore').decode('ascii') , image_url
#     except:
#         return "Could not find the main content of the post." , image_url

def scrape_linkedin_post(url, driver):
    # Open the LinkedIn post URL
    driver.get(url)
    
    # Initialize image_url list
    image_urls = []
    
    try:
        # Wait for all images to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, 'img'))
        )

        # Get all image elements within the post container
        img_elements = driver.find_elements(By.TAG_NAME, 'img')
        
        # Extract URLs from all images
        image_urls = [img.get_attribute('src') for img in img_elements]
        
    except NoSuchElementException:
        image_urls = None
    
    # Wait for the content to load
    try:
        post_content = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'feed-shared-update-v2__description'))
        )
        post_text = post_content.text.encode('ascii', 'ignore').decode('ascii')
    except:
        post_text = "Could not find the main content of the post."

    return post_text, image_urls


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, batch_size=100):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_embeddings = []
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch)
        text_embeddings.extend(zip(batch, batch_embeddings))
    
    vector_store = FAISS.from_embeddings(text_embeddings, embedding=embeddings)
    vector_store.save_local("faiss_index_DS")
    return vector_store

class Document:
    def __init__(self, page_content):
        self.page_content = page_content

# Initialize an empty list to store conversation history
conversation_history = []

# def user_input1(user_question):
#     prompt_template = """
#     Answer the Question from the stack exchange answers given in the context . Explain in detail.
#     Context:\n{context}?\n
#     Question:\n{question}.\n
#     Answer:
#     """
    
#     language, confidence = langid.classify(user_question)
#     print(language)
#     if(language != 'en'):
#         user_question = translate(user_question , language , "en")
#     print(user_question)
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Model for creating vector embeddings
#     new_db = FAISS.load_local("faiss_index_stackexchange", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db
#     docs = new_db.similarity_search(query=user_question, k = 5)
#     # mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db.as_retriever(search_kwargs={'k': 1}) , llm =  model)
#     # docs = mq_retriever.get_relevant_documents(query=user_question)
#     # Regular expression to find the last URL
#     page_content = docs[0].page_content
#     # Find all URLs in the page_content
#     # urls = re.findall(r'https?://\S+', page_content)
#     pattern = r"Link1 : (.*?) \n Link2 : (.*?)$"
#     match = re.search(pattern, page_content, re.MULTILINE)
#     post_link1 = match.group(1)
#     post_link2 = match.group(2)

#     # Get the last URL from the list
#     # image_address = urls[-1] if urls else None
#     # post_link1 = urls[-1]
#     # post_link2 = urls[-2]
#     # print(post_link1)
#     # print(post_link2)
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     return response, None , post_link1 , language

def user_input1(user_question):
    prompt_template = """
    Give answers according to the question asked according to to the transcripts data in the context. The context is about the MMM(Marketing Mix Modelling) workshop.\n
    Also please write the date from the File name from the transcripts data paragraph from which answer is most relatable .\n 
    Also please write the time from the context from the transcripts data paragraph from which answer is most relatable .\n
    Wite date and time in next line after the response .  
    Context:\n{context}?\n
    Question:\n{question}. + Explain in detail.\n
    Answer:
    """
    # New code
    language, confidence = langid.classify(user_question)
    print(language)
    if(language != "en"):
        user_question = translate(user_question , language , "en")
    # New code
    print(user_question)

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    # model = ChatOllama(model='deepseek-r1:8b', temperature=0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Model for creating vector embeddings
    # new_db = FAISS.load_local("faiss_index_images", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db
    new_db1 = FAISS.load_local("Faiss_Index_MMM_workshop1", embeddings,allow_dangerous_deserialization=True)
    # new_db1.merge_from(new_db)
    mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db1.as_retriever(search_kwargs={'k': 5}) , llm =  model)
    q = preprocess_text(user_question)
    docs = mq_retriever.get_relevant_documents(query = q)
    print(docs)
    # Regular expression to find the last URL
    page_content = docs[0].page_content
    # Find all URLs in the page_content
    # urls = re.findall(r'https?://\S+', page_content)

    # Get the last URL from the list
    # image_address = urls[-1] if urls else None
    # post_link = urls[0]
    image_address = find_most_relevant_image(user_question , 'Image_clarifications.xlsx')
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response, image_address , None , language


def user_input(user_question):
    prompt_template = """
    Give answers according to the question asked according to to the transcripts data in the context. The context is about the MMM(Marketing Mix Modelling) workshop.\n
    Also please write the date from the File name from the transcripts data paragraph from which answer is most relatable .\n 
    Also please write the time from the context from the transcripts data paragraph from which answer is most relatable .\n
    Wite date and time in next line after the response .  
    Context:\n{context}?\n
    Question:\n{question}. + Explain in detail.\n
    Answer:
    """
    # New code
    language, confidence = langid.classify(user_question)
    print(language)
    if(language != "en"):
        user_question = translate(user_question , language , "en")
    # New code
    print(user_question)

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Model for creating vector embeddings
    # new_db = FAISS.load_local("faiss_index_images", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db
    new_db1 = FAISS.load_local("Faiss_Index_MMM_workshop", embeddings,allow_dangerous_deserialization=True)
    # new_db1.merge_from(new_db)
    mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db1.as_retriever(search_kwargs={'k': 5}) , llm =  model)
    q = preprocess_text(user_question)
    docs = mq_retriever.get_relevant_documents(query=q)
    print(docs)
    # Regular expression to find the last URL
    page_content = docs[0].page_content
    # Find all URLs in the page_content
    # urls = re.findall(r'https?://\S+', page_content)

    # Get the last URL from the list
    # image_address = urls[-1] if urls else None
    # post_link = urls[0]
    image_address = find_most_relevant_image(user_question , 'Image_clarifications.xlsx')
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response, image_address , None , language

def user_input2(user_question):
    prompt_template = """
    Give answers according to the question asked according to to the transcripts data in the context. The context is about the MMM(Marketing Mix Modelling) workshop.\n
    Also please write the date from the File name from the transcripts data paragraph from which answer is most relatable .\n 
    Also please write the time from the context from the transcripts data paragraph from which answer is most relatable .\n
    Wite date and time in next line after the response .  
    Context:\n{context}?\n
    Question:\n{question}. + Explain in detail.\n
    Answer:
    """
    # New code
    language, confidence = langid.classify(user_question)
    print(language)
    if(language != "en"):
        user_question = translate(user_question , language , "en")
    # New code
    print(user_question)

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    # model = ChatOllama(model='deepseek-r1:8b', temperature=0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Model for creating vector embeddings
    # new_db = FAISS.load_local("faiss_index_images", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db
    new_db1 = FAISS.load_local("Faiss_Index_MMM_workshop2", embeddings,allow_dangerous_deserialization=True)
    # new_db1.merge_from(new_db)
    mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db1.as_retriever(search_kwargs={'k': 5}) , llm =  model)
    q = preprocess_text(user_question)
    docs = mq_retriever.get_relevant_documents(query = q)
    print(docs)
    # Regular expression to find the last URL
    page_content = docs[0].page_content
    # Find all URLs in the page_content
    # urls = re.findall(r'https?://\S+', page_content)

    # Get the last URL from the list
    # image_address = urls[-1] if urls else None
    # post_link = urls[0]
    image_address = find_most_relevant_image(user_question , 'Image_clarifications.xlsx')
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response, image_address , None , language


def user_input3(user_question):
    prompt_template = """
    Give answers according to the question asked according to to the transcripts data in the context. The context is about the MMM(Marketing Mix Modelling) workshop.\n
    Also please write the date from the File name from the transcripts data paragraph from which answer is most relatable .\n 
    Also please write the time from the context from the transcripts data paragraph from which answer is most relatable .\n
    Wite date and time in next line after the response .  
    date is 31st January 2024 and time you can find in the context.\n
    Context:\n{context}?\n
    Question:\n{question}. + Explain in detail.\n
    Answer:
    """
    # New code
    language, confidence = langid.classify(user_question)
    print(language)
    if(language != "en"):
        user_question = translate(user_question , language , "en")
    # New code
    print(user_question)

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    # model = ChatOllama(model='deepseek-r1:8b', temperature=0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Model for creating vector embeddings
    # new_db = FAISS.load_local("faiss_index_images", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db
    new_db1 = FAISS.load_local("Faiss_Index_Tony_Masterclass", embeddings,allow_dangerous_deserialization=True)
    # new_db1.merge_from(new_db)
    mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db1.as_retriever(search_kwargs={'k': 5}) , llm =  model)
    q = preprocess_text(user_question)
    docs = mq_retriever.get_relevant_documents(query = q)
    print(docs)
    # Regular expression to find the last URL
    page_content = docs[0].page_content
    # Find all URLs in the page_content
    # urls = re.findall(r'https?://\S+', page_content)

    # Get the last URL from the list
    # image_address = urls[-1] if urls else None
    # post_link = urls[0]
    image_address = find_most_relevant_image(user_question , 'Image_clarifications.xlsx')
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response, image_address , None , language



def extract_links(pdf_path):
    links = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Extract links using PyMuPDF
        for link in page.get_links():
            if 'uri' in link:
                links.append(link['uri'])
        
        # If OCR is needed
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))

    return links



from selenium.common.exceptions import StaleElementReferenceException

# Function to extract the URL of the next image in the sequence
def extract_next_image_url(post_url,driver):
    driver.get(post_url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.TAG_NAME, 'img'))
    )
    
    try:
        # Update the selector to match the LinkedIn post container
        post_container = driver.find_element(By.CSS_SELECTOR, 'div.feed-shared-update-v2')  # Update the selector here
        
        # Get all image elements within the post container
        img_elements = post_container.find_elements(By.TAG_NAME, 'img')
        
        if len(img_elements) < 2:
            print("Not enough images found")
            return None
        
        # Identify the current image (for example, the first one in the list)
        current_img = img_elements[0]  # Update index if needed
        current_img_src = current_img.get_attribute('src')
        
        # Find the next image in the sequence
        next_img = img_elements[1]  # Adjust index as needed
        next_img_src = next_img.get_attribute('src')
        
        return next_img_src
    except NoSuchElementException:
        print("Element not found")
        return None

def load_in_db():
    url_text_chunks = []
    links = extract_links('List of my best posts -2021.pdf') + extract_links('List of my best posts 2022.pdf') + extract_links('List of my best posts 2023.pdf')
    # Set up the WebDriver (make sure chromedriver is in your PATH or provide the path to the executable)
    driver = webdriver.Chrome()

    linkedin_email = "shauryamishra120210@gmail.com"
    linkedin_password = "Mishra@123"
    # Log in to LinkedIn
    linkedin_login(linkedin_email, linkedin_password , driver)
    # linkedin_post_url = "https://www.linkedin.com/posts/ridhima-kumar7_marketingmixmodeling-marketingattribution-activity-7125811575931760640-Sx65?utm_source=share&utm_medium=member_desktop"

    # file_path = 'MMMGPT_linkedin_blogs.xlsx'
    # df = pd.read_excel(file_path, header=None)
    # links = df.iloc[:, 0].tolist()
    # print(links)
    for linkedin_post_url in links:
        post_text,image_address = scrape_linkedin_post(linkedin_post_url , driver)
        text_chunks = get_text_chunks(post_text)
        # image_address = extract_next_image_url(linkedin_post_url , driver)
        for chunk in text_chunks:
            url_text_chunks.append(f"Linkedin Link : {linkedin_post_url}\n{chunk}\n{image_address}")
    
    new_db1 = get_vector_store(url_text_chunks)

def main():
    load_in_db()

if __name__ == "__main__":
    main()
