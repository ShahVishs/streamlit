import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
import pandas as pd
import os
from streamlit_chat import message
import csv

# # st.image("socialai.jpg")

# Set OpenAI API key
user_api_key = os.environ.get('')

# Load CSV data
csv_file_path = r"C:\Users\shahs\OneDrive\Desktop\strem_ai\2013_Inventory.csv"
data = pd.read_csv(csv_file_path)
texts = data.astype(str).agg(" ".join, axis=1).tolist()

# Set up OpenAI embeddings and create vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", k=8)

# Create the conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo-16k'),
    retriever=retriever
)

# Streamlit UI setup
st.info("Introducing Engage.ai, your cutting-edge partner in streamlining dealership and customer-related operations. ...")  


# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'past' not in st.session_state:
    st.session_state.past = []

# Initialize user name in session state
if 'user_name' not in st.session_state:
    st.session_state.user_name = None

def save_chat_to_csv(user_name, user_input, output):
    with open("conversation_history.csv", mode="a", newline="") as csvfile:
        fieldnames = ["user_name", "question", "answer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:  # Check if the file is empty and write the header
            writer.writeheader()
        writer.writerow({"user_name": user_name, "question": user_input, "answer": output})

# Initialize conversation history with intro_prompt
template_string = """You are a business development manager role \
working in a {industry} you get a text enquiry that is delimited by triple backticks \
You should answer in a style that is {style}. \
text: ```{text}```
"""
industry="""car dealer"""
style = """American English \
in a calm and respectful tone
"""
context = """
Hello
"""
final_prompt = template_string.format(style=style, text=context, industry=industry) 

response_container = st.container()
container = st.container()
chat_history = []  

def conversational_chat(query):
    result = qa({"question": query, "chat_history": chat_history})
    st.session_state.history.append((query, result["answer"]))
    return result["answer"]


with container:
    
    if st.session_state.user_name is None:
        user_name = st.text_input("Your Name:")
        if user_name:
            st.session_state.user_name = user_name

    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Type your question here (:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)

        # Display conversation history with proper differentiation
        with response_container:
            for i, (query, answer) in enumerate(st.session_state.history):
                message(query, is_user=True, key=f"{i}_user", avatar_style="big-smile")
                message(answer, key=f"{i}_answer", avatar_style="thumbs")

        # Save conversation to CSV along with user name
        if st.session_state.user_name:
            save_chat_to_csv(st.session_state.user_name, user_input, output)