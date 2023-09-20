import os
import json
from airtable import Airtable
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from datetime import datetime
from pytz import timezone
import streamlit as st
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
import langchain
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor

# Set your OpenAI API key here
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Streamlit App Title and Info
st.image("socialai.jpg")
st.info("We're developing cutting-edge conversational AI solutions tailored for automotive retail, aiming to provide advanced products and support. As part of our progress, we're establishing an environment to check offerings and also check our website [engane.ai](https://funnelai.com/). This test application answers questions about Inventory, Business details, Financing, Discounts, and Offers. You can explore and play with the inventory dataset [here](https://github.com/buravelliprasad/streamlit/blob/main/dealer_1_inventry.csv) and the appointment dataset [here](https://github.com/buravelliprasad/streamlit_dynamic_retrieval/blob/main/appointment.csv).")

# Get current date and day of the week
current_date = datetime.today().strftime("%m/%d/%y")
day_of_week = datetime.today().weekday()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
current_day = days[day_of_week]

todays_date = current_date
day_of_the_week = current_day

business_details_text = [
    "working days: all Days except Sunday",
    "working hours: 9 am to 7 pm",
    "Phone: (555) 123-4567",
    "Address: 567 Oak Avenue, Anytown, CA 98765",
    "Email: jessica.smith@example.com",
    "Dealer ship location: [Google Maps Link](https://www.google.com/maps/place/Pine+Belt+Mazda/@40.0835762,-74.1764688,15.63z/data=!4m6!3m5!1s0x89c18327cdc07665:0x23c38c7d1f0c2940!8m2!3d40.0835242!4d-74.1742558!16s%2Fg%2F11hkd1hhhb?entry=ttu)"
]

# Create a retriever for business details
retriever_3 = FAISS.from_texts(business_details_text, OpenAIEmbeddings()).as_retriever()

# Initialize Streamlit session state variables
if 'user_name' not in st.session_state:
    st.session_state.user_name = None

if 'new_session' not in st.session_state:
    st.session_state.new_session = True

if 'refreshing_session' not in st.session_state:
    st.session_state.refreshing_session = False

if 'sessions' not in st.session_state:
    st.session_state.sessions = {}

# Initialize session state variables
st.session_state.user_name_input = ""

def save_chat_session(session_data, session_id):
    session_directory = "chat_sessions"
    session_filename = f"{session_directory}/chat_session_{session_id}.json"
    
    if not os.path.exists(session_directory):
        os.makedirs(session_directory)
    
    session_dict = {
        'user_name': session_data['user_name'],
        'chat_history': session_data['chat_history']
    }
    
    try:
        with open(session_filename, "w") as session_file:
            json.dump(session_dict, session_file)
    except Exception as e:
        st.error(f"An error occurred while saving the chat session: {e}")

def load_previous_sessions():
    previous_sessions = {}
    
    if not os.path.exists("chat_sessions"):
        os.makedirs("chat_sessions")
    
    session_files = os.listdir("chat_sessions")
    
    for session_file in session_files:
        session_filename = os.path.join("chat_sessions", session_file)
        
        session_id = session_file.split("_")[-1].split(".json")[0]
        
        with open(session_filename, "r") as session_file:
            session_data = json.load(session_file)
            previous_sessions[session_id] = session_data
    
    return previous_sessions

# Display user name input field
user_name_input = st.text_input("Your name:", key='user_name_input', value=str(st.session_state.user_name_input))

if user_name_input:
    st.session_state.user_name = user_name_input
    st.session_state.user_name_input = user_name_input

# Handle refreshing session
if st.button("Refresh Session"):
    user_name = st.text_input("Your name:", key='user_name_input', value=st.session_state.user_name_input)

    if user_name:
        st.session_state.new_session = True
        st.session_state.user_name_input = user_name
        st.session_state.chat_history = []
        st.session_state.sessions[user_name] = {
            'user_name': user_name,
            'chat_history': []
        }
    else:
        st.session_state.user_name_input = ""

# Load previous sessions if it's a new session
if st.session_state.new_session:
    st.session_state.sessions = load_previous_sessions()

# Display chat sessions in the sidebar
st.sidebar.header("Chat Sessions")

for session_id, session_data in st.session_state.sessions.items():
    session_key = f"session_{session_id}"
    
    if st.sidebar.button(f"Session {session_id}"):
        st.session_state.chat_history = session_data['chat_history']
        st.session_state.new_session = False

    if session_id == st.session_state.user_name:
        st.session_state.user_name = st.text_input(f"Your name for Session {session_id}:", value=st.session_state.user_name, key=session_key)
        if st.session_state.user_name:
            st.session_state.new_session = False

# Load and create the inventory retriever
file_1 = 'dealer_1_inventry.csv'
loader = CSVLoader(file_path=file_1)
docs_1 = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore_1 = FAISS.from_documents(docs_1, embeddings)
retriever_1 = vectorstore_1.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# Create tools for retrievers
tool1 = create_retriever_tool(
    retriever_1, 
    "search_car_dealership_inventory",
    "Searches and returns documents regarding the car inventory. Input should be a single string."
)

tool3 = create_retriever_tool(
    retriever_3, 
    "search_business_details",
    "Searches and returns documents related to business working days and hours, location, and address details."
)

tools = [tool1, tool3]

# Set up Airtable for data storage
airtable_api_key = st.secrets["AIRTABLE"]["AIRTABLE_API_KEY"]
os.environ["AIRTABLE_API_KEY"] = airtable_api_key
AIRTABLE_BASE_ID = "appAVFD4iKFkBm49q"
AIRTABLE_TABLE_NAME = "Question_Answer_Data"

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'generated' not in st.session_state:
    st.session_state.generated = []

if 'past' not in st.session_state:
    st.session_state.past = []

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
langchain.debug = True
memory_key = "history"
memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

template = (
    """You're the Business Development Manager at our car dealership.
    When responding to inquiries, please adhere to the following guidelines:
    Car Inventory Questions: If the customer's inquiry lacks specific details such as their preferred
    make, model, new or used car, and trade-in, kindly engage by asking for these specifics.
    Specific Car Details: When addressing questions about a particular car, limit the information provided
    to make, year, model, and trim. For example, if asked about 'Do you have Jeep Cherokee Limited 4x4',
    the best answer should be 'Yes, we have Jeep Cherokee Limited 4x4: Year: 2022, Model: [Model], Make: [Make], Trim: [Trim]'.
    Scheduling Appointments: If the customer's inquiry lacks specific details such as their preferred
    day, date, or time, kindly engage by asking for these specifics.
    
    Use today's date and day to find the appointment date from the user's input and check for appointment availability for that specific date and time. 
    If the appointment schedule is not available, provide this link: www.dummy_calenderlink.com to schedule an appointment by the user.
    
    If appointment schedules are not available, you should send this link: www.dummy_calendarlink.com to the 
    customer to schedule an appointment on your own.

    Encourage Dealership Visit: Our goal is to encourage customers to visit the dealership for test drives or
    receive product briefings from our team. After providing essential information on the car's make, model,
    color, and basic features, kindly invite the customer to schedule an appointment for a test drive or visit us
    for a comprehensive product overview by our experts.

    Please maintain a courteous and respectful tone in your American English responses.
    If you're unsure of an answer, respond with 'I am sorry.'
    Make every effort to assist the customer promptly while keeping responses concise, not exceeding two sentences.
    
    Feel free to use any tools available to look up for relevant information.
    Answer the question not more than two sentences."""
)

details = "Today's current date is " + todays_date + " and today's weekday is " + day_of_the_week + "."

input_template = template.format(details=details)

system_message = SystemMessage(
    content=input_template)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
)
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

if 'agent_executor' not in st.session_state:
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, return_intermediate_steps=True)
    st.session_state.agent_executor = agent_executor
else:
    agent_executor = st.session_state.agent_executor

response_container = st.container()
container = st.container()

airtable = Airtable(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, api_key=airtable_api_key)

def save_chat_to_airtable(user_name, user_input, output):
    try:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        airtable.insert(
            {
                "username": user_name,
                "question": user_input,
                "answer": output,
                "timestamp": timestamp,
            }
        )
    except Exception as e:
        st.error(f"An error occurred while saving data to Airtable: {e}")

def conversational_chat(user_input):
    result = agent_executor({"input": user_input})
    st.session_state.chat_history.append((user_input, result["output"]))
    return result["output"]

user_input = ""
output = ""  

with st.form(key='my_form', clear_on_submit=True):
    user_input = st.text_input("Query:", placeholder="Type your question here :)", key='input')
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    output = conversational_chat(user_input)
    st.session_state.chat_history.append((user_input, output))

if st.session_state.user_name and st.session_state.chat_history:
    current_session_data = {
        'user_name': st.session_state.user_name,
        'chat_history': st.session_state.chat_history
    }
    st.session_state.past.append(current_session_data)

with response_container:
    for i, (query, answer) in enumerate(st.session_state.chat_history):
        user_name = st.session_state.user_name
        message(query, is_user=True, key=f"{i}_user", avatar_style="big-smile")
        message(answer, key=f"{i}_answer", avatar_style="thumbs")

    if st.session_state.user_name:
        try:
            save_chat_to_airtable(st.session_state.user_name, user_input, output)
        except Exception as e:
            st.error(f"An error occurred: {e}")
