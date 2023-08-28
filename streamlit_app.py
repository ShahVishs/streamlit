from langchain.llms import OpenAI
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain import PromptTemplate
import tempfile
import pandas as pd
import os
import csv
import gspread
from google.oauth2 import service_account
import base64
from datetime import datetime
from pytz import timezone
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Access PostgreSQL connection details from secrets
db_secrets = st.secrets["postgres"]
db_username = db_secrets["user"]
db_password = db_secrets["password"]
db_host = db_secrets["host"]
db_port = db_secrets["port"]
db_name = db_secrets["dbname"]

# Construct the connection URI
SQLALCHEMY_DATABASE_URI = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(SQLALCHEMY_DATABASE_URI)

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True)
    timestamp = Column(TIMESTAMP)
    user_name = Column(String)
    user_input = Column(String)
    output = Column(String)

# Initialize connection
try:
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    Base.metadata.create_all(engine)
    conn = engine.raw_connection()  # Get the raw connection for psycopg2
except Exception as e:
    st.error(f"An error occurred while connecting to the database: {e}")

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
st.image("socialai.jpg")
file = r'dealer_1_inventry.csv'
loader = CSVLoader(file_path=file)
docs = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", k=8)
# Streamlit UI setup
st.info(" We're developing cutting-edge conversational AI solutions tailored for automotive retail, aiming to provide advanced products and support. As part of our progress, we're establishing a environment to check offerings and also check Our website [engane.ai](https://funnelai.com/). This test application answers about Inventry, Business details, Financing and Discounts and Offers related questions. [here](https://github.com/buravelliprasad/streamlit/blob/main/dealer_1_inventry.csv) is a inventry dataset explore and play with the data.")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'past' not in st.session_state:
    st.session_state.past = []
# Initialize user name in session state
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
    
# def create_db_connection():
#     connection = psycopg2.connect(
#         host="localhost",
#         port=5432,
#         database="smai_local",
#         user="postgres",
#         password="root"
#     )
#     return connection

def save_chat_to_postgresql(user_name, user_input, output, timestamp):
    try:
        # Create a new connection
        conn = engine.raw_connection()
        
        insert_query = "INSERT INTO chat_history (timestamp, user_name, user_input, output) VALUES (%s, %s, %s, %s)"
        data = (timestamp, user_name, user_input, output)

        with conn.cursor() as cur:
            cur.execute(insert_query, data)
            conn.commit()

        # st.success("Data saved to PostgreSQL!")
    except Exception as e:
        st.error(f"Error saving data to PostgreSQL: {str(e)}")
        

    
# Initialize conversation history with intro_prompt
custom_template = """You are a business development manager role \
working in a car dealership you get a text enquiry regarding inventry, business details and finance. 
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. 
At the end of standalone question add this 'You should answer in a style that is American English in a calm and respectful tone.' 
If you do not know the answer reply with 'I am sorry'.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
# Model details
qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo-16k'),
    retriever=retriever,
    condense_question_prompt=CUSTOM_QUESTION_PROMPT
)
response_container = st.container()
container = st.container()
chat_history = []  # Store the conversation history here

def conversational_chat(user_input):
    result = qa({"question": user_input, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((user_input, result["answer"]))
    return result["answer"]

# Streamlit main code
with container:
    if st.session_state.user_name is None:
        user_name = st.text_input("Your name:")
        if user_name:
            st.session_state.user_name = user_name
            
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Type your question here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
    
    if submit_button and user_input:
        output = conversational_chat(user_input)
        utc_now = datetime.now(timezone('UTC'))

        with response_container:
            st.session_state.chat_history.append((user_input, output))  # Add the conversation to chat history
            
            for i, (query, answer) in enumerate(st.session_state.chat_history):
                message(query, is_user=True, key=f"{i}_user", avatar_style="big-smile")
                message(answer, key=f"{i}_answer", avatar_style="thumbs")
    
            if st.session_state.user_name:
                try:
                    save_chat_to_postgresql(st.session_state.user_name, user_input, output, utc_now.strftime('%Y-%m-%d-%H-%M-%S'))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
