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
    "working hours: 9 am to 7 pm",
    "Phone: (555) 123-4567",
    "Address: 567 Oak Avenue, Anytown, CA 98765, Email: jessica.smith@example.com",
    "dealer ship location: https://www.google.com/maps/place/Pine+Belt+Mazda/@40.0835762,-74.1764688,15.63z/data=!4m6!3m5!1s0x89c18327cdc07665:0x23c38c7d1f0c2940!8m2!3d40.0835242!4d-74.1742558!16s%2Fg%2F11hkd1hhhb?entry=ttu"
]
retriever_3 = FAISS.from_texts(business_details_text, OpenAIEmbeddings()).as_retriever()

file_1 = r'dealer_1_inventry.csv'

loader = CSVLoader(file_path=file_1)
docs_1 = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore_1 = FAISS.from_documents(docs_1, embeddings)
retriever_1 = vectorstore_1.as_retriever(search_type="similarity", search_kwargs={"k": 8})  # check without similarity search and k=8

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
airtable = Airtable(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, api_key=airtable_api_key)

# Streamlit UI setup
st.info("We're developing cutting-edge conversational AI solutions tailored for automotive retail, aiming to provide advanced products and support. As part of our progress, we're establishing an environment to check offerings and also check Our website [engane.ai](https://funnelai.com/). This test application answers about Inventory, Business details, Financing and Discounts and Offers related questions. [Here](https://github.com/buravelliprasad/streamlit/blob/main/dealer_1_inventry.csv) is an inventory dataset to explore and play with the data. The appointment dataset is not available in this example.")
# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = {}
if 'user_name' not in st.session_state:
    st.session_state.user_name = None

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
langchain.debug = True
memory_key = "history"
memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)
template = (
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
day, date or time kindly engage by asking for these specifics. {details} Use these details that are today's date and day /
to find the appointment date from the user's input and check for appointment availability for that specific date and time. 
If the appointment schedule is not available, provide this 
link: www.dummy_calenderlink.com to schedule an appointment by the user himself. 
If appointment schedules are not available, you should send this link: www.dummy_calendarlink.com to the 
customer to schedule an appointment on your own.

Encourage Dealership Visit: Our goal is to encourage customers to visit the dealership for test drives or/
receive product briefings from our team. After providing essential information on the car's make, model,/
color, and basic features, kindly invite the customer to schedule an appointment for a test drive or visit us/
for a comprehensive product overview by our experts.

Please maintain a courteous and respectful tone in your American English responses./
If you're unsure of an answer, respond with 'I am sorry.'/
Make every effort to assist the customer promptly while keeping responses concise, not exceeding two sentences."
Feel free to use any tools available to look up relevant information.
Answer the question not more than two sentence.""")

details = "Today's current date is " + todays_date + " today's weekday is " + day_of_the_week + "."

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

chat_history = []

def conversational_chat(user_input):
    result = agent_executor({"input": user_input})
    st.session_state.chat_history.append((user_input, result["output"]))
    return result["output"]

# Create a sidebar for displaying chat history
st.sidebar.title("Chat History")

# Create a selectbox to choose a user and display their chat history
selected_user = st.sidebar.selectbox("Select User", st.session_state.chat_histories.keys())
if selected_user:
    st.sidebar.text("Chat History:")
    user_chat_history = st.session_state.chat_histories[selected_user]
    for i, (query, answer) in enumerate(user_chat_history):
        st.sidebar.text(f"{i + 1}. User: {query}")
        st.sidebar.text(f"   Assistant: {answer}")

with container:
    if st.session_state.user_name is None:
        user_name = st.text_input("Your name:")
        if user_name:
            st.session_state.user_name = user_name
            
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Type your question here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
    
    # Create a button to start a new chat session and save the previous chat history to Airtable
    if st.button("Refresh"):
        # Save the current chat history to Airtable
        previous_user = st.session_state.user_name
        previous_chat_history = st.session_state.chat_history
        for query, answer in previous_chat_history:
            save_chat_to_airtable(previous_user, query, answer)
        
        # Reset the chat history
        st.session_state.chat_history = []
        user_input = ""  # Clear the input field

    if submit_button and user_input:
        output = conversational_chat(user_input)
        session_key = f"{st.session_state.user_name}_{todays_date}"
        if session_key not in st.session_state.chat_histories:
            st.session_state.chat_histories[session_key] = []
        st.session_state.chat_histories[session_key].append((user_input, output))

# Display Chat Histories
with response_container:
    session_key = f"{st.session_state.user_name}_{todays_date}"
    chat_history = st.session_state.chat_histories.get(session_key, [])
       
    for i, (query, answer) in enumerate(chat_history):
        message(query, is_user=True, key=f"{i}_user", avatar_style="big-smile")
        message(answer, key=f"{i}_answer", avatar_style="thumbs")
