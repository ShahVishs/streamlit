import os
from airtable import Airtable
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from pytz import timezone
from datetime import datetime
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
import langchain
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
import json
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
st.image("socialai.jpg")

# datetime.datetime.now()
datetime.now()
# Get the current date in "%m/%d/%y" format
# current_date = datetime.date.today().strftime("%m/%d/%y")
current_date = datetime.today().strftime("%m/%d/%y")
# Get the day of the week (0: Monday, 1: Tuesday, ..., 6: Sunday)
# day_of_week = datetime.date.today().weekday()
day_of_week = datetime.today().weekday()
# Convert the day of the week to a string representation
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
current_day = days[day_of_week]

# print("Current date:", current_date)
# print("Current day:", current_day)
todays_date = current_date
day_of_the_week = current_day

business_details_text = [
    "working days: all Days except sunday",
    "working hours: 9 am to 7 pm"
    "Phone: (555) 123-4567"
    "Address: 567 Oak Avenue, Anytown, CA 98765, Email: jessica.smith@example.com",
    "dealer ship location: https://www.google.com/maps/place/Pine+Belt+Mazda/@40.0835762,-74.1764688,15.63z/data=!4m6!3m5!1s0x89c18327cdc07665:0x23c38c7d1f0c2940!8m2!3d40.0835242!4d-74.1742558!16s%2Fg%2F11hkd1hhhb?entry=ttu"
]
retriever_3 = FAISS.from_texts(business_details_text, OpenAIEmbeddings()).as_retriever()
# Initialize session state
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'sessions' not in st.session_state:
    st.session_state.sessions = {}

# Initialize a flag to track whether the session is new or switching to a previous session
if 'new_session' not in st.session_state:
    st.session_state.new_session = True

# Function to save the current chat session
def save_chat_session(session_data, session_id):
    session_filename = f"chat_sessions/chat_session_{session_id}.json"
    
    # Convert session_data to a dictionary
    session_dict = {
        'user_name': session_data['user_name'],
        'chat_history': session_data['chat_history']
    }
    
    with open(session_filename, "w") as session_file:
        json.dump(session_dict, session_file)

# Function to load previous chat sessions from files
def load_previous_sessions():
    previous_sessions = {}
    
    # Check if the chat_sessions folder exists
    if not os.path.exists("chat_sessions"):
        os.makedirs("chat_sessions")
    
    # Get a list of chat session files
    session_files = os.listdir("chat_sessions")
    
    # Load each chat session file
    for session_file in session_files:
        session_filename = os.path.join("chat_sessions", session_file)
        
        # Extract session ID from the file name
        session_id = session_file.split("_")[-1].split(".json")[0]
        
        with open(session_filename, "r") as session_file:
            session_data = json.load(session_file)
            previous_sessions[session_id] = session_data
    
    return previous_sessions

# Create a Streamlit button for starting a new session
if st.button("Refresh Session"):
    # Modify the logic for setting the user's name
    if 'user_name' not in st.session_state or st.session_state.new_session:
        user_name = st.text_input("Your name:", key='user_name_input', value=st.session_state.user_name)
        if user_name:
            st.session_state.user_name = user_name
            st.session_state.new_session = False  # Mark that it's not a new session
    
    # Save the current session and start a new one
    current_session = {
        'user_name': st.session_state.user_name,
        'chat_history': st.session_state.chat_history
    }
    
    # Generate a unique session_id based on the timestamp
    session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
    save_chat_session(current_session, session_id)

    # Clear session state variables to start a new session
    st.session_state.chat_history = []

# Load previous chat sessions
st.session_state.sessions = load_previous_sessions()

# Display a list of session names in the sidebar
st.sidebar.header("Chat Sessions")

for session_id in st.session_state.sessions.keys():
    if st.sidebar.button(f"Session {session_id}"):
        # When a session ID is clicked, update the chat history to show messages for that session
        selected_session_data = st.session_state.sessions[session_id]
        st.session_state.chat_history = selected_session_data['chat_history']
        st.session_state.new_session = False  # Mark that it's not a new session

file_1 = r'dealer_1_inventry.csv'

loader = CSVLoader(file_path=file_1)
docs_1 = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore_1 = FAISS.from_documents(docs_1, embeddings)
retriever_1 = vectorstore_1.as_retriever(search_type="similarity", search_kwargs={"k": 8})#check without similarity search and k=8

# Create the first tool
tool1 = create_retriever_tool(
    retriever_1, 
     "search_car_dealership_inventory",
     "Searches and returns documents regarding the car inventory and Input should be a single string strictly."
)

# Create the third tool
tool3 = create_retriever_tool(
    retriever_3, 
    "search_business_details",
    "Searches and returns documents related to business working days and hours, location and address details."
)

# Append all tools to the tools list
tools = [tool1, tool3]

# airtable
airtable_api_key = st.secrets["AIRTABLE"]["AIRTABLE_API_KEY"]
os.environ["AIRTABLE_API_KEY"] = airtable_api_key
AIRTABLE_BASE_ID = "appAVFD4iKFkBm49q"  
AIRTABLE_TABLE_NAME = "Question_Answer_Data" 
# Streamlit UI setup
st.info(" We're developing cutting-edge conversational AI solutions tailored for automotive retail, aiming to provide advanced products and support. As part of our progress, we're establishing a environment to check offerings and also check Our website [engane.ai](https://funnelai.com/). This test application answers about Inventry, Business details, Financing and Discounts and Offers related questions. [here](https://github.com/buravelliprasad/streamlit/blob/main/dealer_1_inventry.csv) is a inventry dataset explore and play with the data. Appointment dataset [here](https://github.com/buravelliprasad/streamlit_dynamic_retrieval/blob/main/appointment.csv)")
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

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature = 0)
langchain.debug=True
memory_key = "history"
memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)
template=(
"""You're the Business Development Manager at our car dealership./
When responding to inquiries, please adhere to the following guidelines:
Car Inventory Questions: If the customer's inquiry lacks specific details such as their preferred/
make, model, new or used car, and trade-in, kindly engage by asking for these specifics./
Specific Car Details: When addressing questions about a particular car, limit the information provided/
to make, year, model, and trim. For example, if asked about 
'Do you have Jeep Cherokee Limited 4x4'
Best answer should be 'Yes we have,
Jeep Cherokee Limited 4x4:
Year: 2022
Model :
Make :
Trim:
scheduling Appointments: If the customer's inquiry lacks specific details such as their preferred/
day, date or time kindly engage by asking for these specifics. {details} Use these details that is todays date and day /
to find the appointment date from the users input and check for appointment availabity for that specific date and time. 
If the appointment schedule is not available provide this 
link: www.dummy_calenderlink.com to schedule appointment by the user himself. 
If appointment schedules are not available, you should send this link: www.dummy_calendarlink.com to the 
costumer to schedule an appointment on your own.

Encourage Dealership Visit: Our goal is to encourage customers to visit the dealership for test drives or/
receive product briefings from our team. After providing essential information on the car's make, model,/
color, and basic features, kindly invite the customer to schedule an appointment for a test drive or visit us/
for a comprehensive product overview by our experts.

Please maintain a courteous and respectful tone in your American English responses./
If you're unsure of an answer, respond with 'I am sorry.'/
Make every effort to assist the customer promptly while keeping responses concise, not exceeding two sentences."
Feel free to use any tools available to look up for relevant information.
Answer the question not more than two sentence.""")

details = "Today's current date is " + todays_date + " and today's week day is " + day_of_the_week + "."

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

if st.session_state.user_name is None:
    user_name = st.text_input("Your name:")
    if user_name:
        st.session_state.user_name = user_name
        st.session_state.name_entered = True

user_input = ""
with st.form(key='my_form', clear_on_submit=True):
    user_input = st.text_input("Query:", placeholder="Type your question here (:", key='input')
    if user_input:
        st.write(f"**User:** {user_input}")
        # Add the user's question to the current session's chat history
        st.session_state.chat_history.append((user_input, "AI's response here."))
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    output = conversational_chat(user_input)
    st.session_state.chat_history.append((user_input, output))

# Save the current session data to past sessions
if st.session_state.user_name and st.session_state.chat_history:
    current_session_data = {
        'user_name': st.session_state.user_name,
        'chat_history': st.session_state.chat_history
    }
    st.session_state.past.append(current_session_data)

with response_container:
    for i, (query, answer) in enumerate(st.session_state.chat_history):
        message(query, is_user=True, key=f"{i}_user", avatar_style="big-smile")
        message(answer, key=f"{i}_answer", avatar_style="thumbs")

    if st.session_state.user_name:
        try:
            save_chat_to_airtable(st.session_state.user_name, user_input, output)
        except Exception as e:
            st.error(f"An error occurred: {e}")
