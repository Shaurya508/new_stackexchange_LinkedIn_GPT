import streamlit as st
import pandas as pd
from new import user_input , user_input1
from io import BytesIO
from PIL import Image , UnidentifiedImageError
import requests
from trial import translate
import re
from Levenshtein import distance as levenshtein_distance

# Define the maximum number of free queries
QUERY_LIMIT = 100

# Initialize session state for tracking the number of queries, conversation history, suggested questions, and authentication
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'suggested_question' not in st.session_state:
    st.session_state.suggested_question = ""

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'generate_response' not in st.session_state:
    st.session_state.generate_response = False

if 'chat' not in st.session_state:
    st.session_state.chat = ""

def clean_text(text):
    # Remove asterisks used for bold formatting
    text = re.sub(r'\*+', '', text)
    # Remove text starting from "For more details"
    text = re.sub(r'For more details.*$', '', text, flags=re.IGNORECASE)
    return text

def authenticate_user(email):
    # Load the Excel file
    df = pd.read_excel('user.xlsx')
    # Convert the input email to lowercase
    email = email.lower()
    # Convert the emails in the dataframe to lowercase
    df['Email'] = df['Email'].str.lower()
    # Check if the email matches any entry in the file
    user = df[df['Email'] == email]
    if not user.empty:
        return True
    return False

def get_image_link(article_link, file_path='Linkidin_blogs.xlsx'):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Ensure the columns are named correctly
    df.columns = ['Article Link', 'Image link']

    # Create a dictionary mapping article links to image links
    link_mapping = dict(zip(df['Article Link'], df['Image link']))

    # Find the most similar article link using Levenshtein distance
    most_similar_link = min(df['Article Link'], key=lambda x: levenshtein_distance(x, article_link))
    image_link = link_mapping.get(most_similar_link, "Image link not found")
    
    if image_link == "Image link not found" or image_link == 0:
        return None
    return image_link

def create_ui():
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    footer:after {content:''; display:block; position:relative; top:2px; color: transparent; background-color: transparent;}
    .viewerBadge_container__1QSob {display: none !important;}
    .stActionButton {display: none !important;}
    ::-webkit-scrollbar {
        width: 12px;  /* Keep the width of the scrollbar */
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    .scroll-icon {
        position: fixed;
        bottom: 40px;  /* Adjusted the position upwards */
        right: 150px;
        font-size: 32px;
        cursor: pointer;
        color: #0adbfc;
        z-index: 1000;
    }
    </style>
    <script>
    function scrollToBottom() {
        window.scrollTo(0, 50000);
    }
    </script>
    """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #0adbfc;'><u> Venkat's LinkedIn GPT</u></h2>", unsafe_allow_html=True)
    st.sidebar.image("Aryma Labs Logo.jpeg")
    st.sidebar.markdown("<h3 style='color: #08daff;'>Welcome to Venkat's LinkedIn GPT</h2>", unsafe_allow_html=True)
    # st.sidebar.write("Ask anything about MMM and get accurate answers.")
    

    if not st.session_state.authenticated:
        st.markdown("<h3 style='color: #4682B4;'>Login</h3>", unsafe_allow_html=True)
        with st.form(key='login_form'):
            email = st.text_input("Email")
            login_button = st.form_submit_button(label='Login')

            if login_button:
                if authenticate_user(email):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid email or password. Please try again.")
        return

    # st.sidebar.markdown("<h5 style='color: #08daff;'>Popular Questions</h3>", unsafe_allow_html=True)

    # suggested_questions = [
    #     "What is Market Mix modelling?",
    #     "What are Contribution Charts?",
    #     "Provide code examples from Robyn.",
    #     "How MMMs can be calibrated and validated?",
    #     "Why Frequentist MMM is better than Bayesian MMM?"
    # ]

    # for i, question in enumerate(suggested_questions):
    #     if st.sidebar.button(question, key=f"button_{i}", use_container_width=True):
    #         st.session_state.suggested_question = question
    #         st.session_state.generate_response = True

    # Display the conversation history in reverse order to resemble a chat interface
    chat_container = st.container()
    LANGUAGES = {
    'Arabic': 'ar',
    'Azerbaijani': 'az',
    'Catalan': 'ca',
    'Chinese': 'zh',
    'Czech': 'cs',
    'Danish': 'da',
    'Dutch': 'nl',
    'English': 'en',
    'Esperanto': 'eo',
    'Finnish': 'fi',
    'French': 'fr',
    'German': 'de',
    'Greek': 'el',
    'Hebrew': 'he',
    'Hindi': 'hi',
    'Hungarian': 'hu',
    'Indonesian': 'id',
    'Irish': 'ga',
    'Italian': 'it',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Persian': 'fa',
    'Polish': 'pl',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Slovak': 'sk',
    'Spanish': 'es',
    'Swedish': 'sv',
    'Turkish': 'tr',
    'Ukrainian': 'uk',
    'bengali' : 'bn'
}
    with chat_container:
        if st.session_state.conversation_history == []:
            col1, col2 = st.columns([1, 8])
            with col1:
                st.image('download.png', width=30)
            with col2:
                
                st.write("Hello, I am Venkat's LinkedIn GPT . How can I help you?")
    for idx , (q, r , url , post_link , language) in enumerate(st.session_state.conversation_history):
        st.markdown(f"<p style='text-align: right; color: #484f4f;'><b>{q}</b></p>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 8])
        r1 = r
        with col1:
            st.image('download.png', width=30)
        with col2:
            if(language != "en"):
                r = translate(clean_text(r) , "en" , language)
            st.write(r + "\n")
            if(language != "en"):
                st.write(translate("For more details, please visit", from_lang='en', to_lang= language) + ": " + post_link)
            else:
                st.write("For more details, please visit :" + post_link)
            target_language = st.selectbox('Select target language', options=list(LANGUAGES.keys()), key=f'target_language_{idx}')
        if st.button('Translate', key=f'translate_button_{idx}'):
    
            if target_language:
                # Translation
                if(target_language == "English"):
                    st.write(r1)
                    st.write("For more details, please visit : " + post_link)
                else:
                    if(language != LANGUAGES[target_language]):
                        # print(language + "\n" + LANGUAGES[target_language])
                        translated_text = translate(clean_text(r1), from_lang= "en" , to_lang=LANGUAGES[target_language])
                        added_text = translate("For more details, please visit", from_lang= "en", to_lang=LANGUAGES[target_language])
                    else:
                        translated_text = clean_text(r)
                        if(language != "en"):
                            added_text = translate("For more details, please visit", from_lang= "en", to_lang= language)
                        else:
                            added_text = "For more details, please visit"
                # Display the translation
                # st.subheader('Translated Text')
                    st.write( translated_text + "\n\n" + added_text + ": " + post_link)
        image_link = get_image_link(post_link)
        if image_link is not None and url is not None:
            try:
                response = requests.get(image_link)
                img = Image.open(BytesIO(response.content))
                st.image(img, use_column_width=True)
            except UnidentifiedImageError:
                pass
                    
    # Get user input at the bottom
    st.markdown("---")
    instr = "Ask a question:"
    with st.form(key='input_form', clear_on_submit=True):
        # Text input box taking the full width
        question = st.text_input(
            instr,
            value=st.session_state.get('suggested_question', ''),
            key="input_question",
            placeholder=instr,
            label_visibility='collapsed'
        )
        
        # Buttons placed below the text box
        col1, col2 = st.columns(2)
        col1, col2 = st.columns([1,2], gap="small")
        with col1:
            linkedin_button = st.form_submit_button(label='Chat with LinkedIn posts')
    
        if linkedin_button and question:
            st.session_state.generate_response = 'Chat with MMM workshop'
        # elif stackexchange_button and question:
        #     st.session_state.generate_response = 'stackexchange'
    
    if st.session_state.generate_response and question:
        if st.session_state.query_count >= QUERY_LIMIT:
            st.warning("You have reached the limit of free queries. Please consider our pricing options for further use.")
        else:
            with st.spinner("Generating response..."):
                if st.session_state.generate_response == 'linkedin':
                    response, image_address, post_link, language = user_input(question)
                # elif st.session_state.generate_response == 'stackexchange':
                #     response, image_address, post_link ,language = user_input1(question)
                    
                output_text = response.get('output_text', 'No response')  # Extract the 'output_text' from the response
                st.session_state.chat += str(output_text)
                st.session_state.conversation_history.append((question, output_text, image_address, post_link , language))
                st.session_state.suggested_question = ""  # Reset the suggested question after submission
                st.session_state.query_count += 1  # Increment the query count
                st.session_state.generate_response = False
                st.rerun()
    
    # col1 = st.columns([1], gap="small")[0]  # Single column layout for the button
    # with col1:
    # workshop_button = st.form_submit_button(label='Chat with MMM workshop')

    # if workshop_button and question:
    #     st.session_state.generate_response = 'mmm_workshop'

    # if st.session_state.generate_response and question:
    #     if st.session_state.query_count >= QUERY_LIMIT:
    #         st.warning("You have reached the limit of free queries. Please consider our pricing options for further use.")
    #     else:
    #         with st.spinner("Generating response..."):
    #             # Call the user_input function for the MMM workshop
    #             response, image_address, post_link, language = user_input(question)
            
    #             output_text = response.get('output_text', 'No response')  # Extract the 'output_text' from the response
    #             st.session_state.chat += str(output_text)
    #             st.session_state.conversation_history.append((question, output_text, image_address, post_link, language))
    #             st.session_state.suggested_question = ""  # Reset the suggested question after submission
    #             st.session_state.query_count += 1  # Increment the query count
    #             st.session_state.generate_response = False
    #             st.rerun()



    # Scroll to bottom icon
    st.markdown("""
        <div class="scroll-icon">⬇️</div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #A9A9A9;'>Powered by: Aryma Labs</p>", unsafe_allow_html=True)

# Main function to run the app
def main():
    create_ui()

if __name__ == "__main__":
    main()
