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

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Display an image or logo
st.image("socialai.jpg")

# Get the current date and day of the week
current_date = datetime.today().strftime("%m/%d/%y")
day_of_week = datetime.today().strftime("%A")

# Define business details for retrieval
business_details_text = [
    "Working days: All days except Sunday",
    "Working hours: 9 am to 7 pm",
    "Phone: (555) 123-4567",
    "Address: 567 Oak Avenue, Anytown, CA 98765",
    "Email: jessica.smith@example.com",
    "Dealer ship location: https://www.google.com/maps/place/Pine+Belt+Mazda/"
]

# Create a retriever for business details
retriever_3 = FAISS.from_texts(business_details_text, OpenAIEmbeddings()).as_retriever()
file_1 = r'dealer_1_inventry.csv'

loader = CSVLoader(file_path=file_1)
docs_1 = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore_1 = FAISS.from_documents(docs_1, embeddings)
retriever_1 = vectorstore_1.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# Define the tools for retrieval
tool1 = create_retriever_tool(
    retriever_1, 
     "search_car_dealership_inventory",
     "Searches and returns documents regarding the car inventory. Input should be a single string."
)

tool3 = create_retriever_tool(
    retriever_3, 
    "search_business_details",
    "Searches and returns documents related to business working days, hours, location, and address details."
)

tools = [tool1, tool3]

# Set up Airtable integration
airtable_api_key = st.secrets["AIRTABLE"]["AIRTABLE_API_KEY"]
os.environ["AIRTABLE_API_KEY"] = airtable_api_key
AIRTABLE_BASE_ID = "appAVFD4iKFkBm49q"  
AIRTABLE_TABLE_NAME = "Question_Answer_Data" 

# Streamlit UI setup
st.info("We're developing cutting-edge conversational AI solutions tailored for automotive retail, aiming to provide advanced products and support. As part of our progress, we're establishing an environment to check offerings and also check our website [engane.ai](https://funnelai.com/). This test application answers questions about inventory, business details, financing, discounts, and offers. You can explore and play with the inventory dataset [here](https://github.com/buravelliprasad/streamlit/blob/main/dealer_1_inventory.csv). The appointment dataset is [here](https://github.com/buravelliprasad/streamlit_dynamic_retrieval/blob/main/appointment.csv).")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'past' not in st.session_state:
    st.session_state.past = []
# Initialize chat histories
if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = {}

# Initialize user name in session state
if 'user_name' not in st.session_state:
    st.session_state.user_name = None

# Initialize Language Model
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
langchain.debug = True
memory_key = "history"
memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

# Define the template for system message
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
Answer the question not more than two sentence."""
)

# Create the details string
details = f"Today's current date is {current_date} and today's week day is {day_of_week}."

# Create the input template for the assistant
input_template = template.format(details=details)

# Create a system message with the input template
system_message = SystemMessage(content=input_template)

# Create a prompt with the system message
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
)

# Create an agent with the language model, tools, and prompt
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
if 'agent_executor' not in st.session_state:
	agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, return_intermediate_steps=True)
	st.session_state.agent_executor = agent_executor
else:
	agent_executor = st.session_state.agent_executor

response_container = st.container()
container = st.container()
# Define a function to save chat history to Airtable
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
chat_history=[]
# Define a function for conversational chat
def conversational_chat(user_input):
    result = agent_executor({"input": user_input})
    st.session_state.chat_history.append((user_input, result["output"]))
    return result["output"]

# If the user name is not set, prompt the user to enter their name
with container:
    if st.session_state.user_name is None:
        user_name = st.text_input("Your name:")
        if user_name:
            st.session_state.user_name = user_name

    # Create a form for user input
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Type your question here (:")
        submit_button = st.form_submit_button(label='Send')

    # If the submit button is clicked and user input is provided
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
	       # utc_now = datetime.utcnow()
   
       with response_container:
           for i, (query, answer) in enumerate(st.session_state.chat_history):
               message(query, is_user=True, key=f"{i}_user", avatar_style="big-smile")
               message(answer, key=f"{i}_answer", avatar_style="thumbs")
   
           if st.session_state.user_name:
               try:
                   save_chat_to_airtable(st.session_state.user_name, user_input, output)
               except Exception as e:
                   st.error(f"An error occurred: {e}")


