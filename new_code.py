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
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
st.image("socialai.jpg")
file = r'dealer_1_inventry.csv'
loader = CSVLoader(file_path=file)
docs = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", k=8)
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
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. At the end of standalone question add this 'Answer the question in english language.' If you do not know the answer reply with 'I am sorry'.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo-16k'),
    retriever=retriever,
    condense_question_prompt=CUSTOM_QUESTION_PROMPT
#     return_source_documents=True
)
response_container = st.container()
container = st.container()
chat_history=[] 
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

