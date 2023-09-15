import os
from airtable import Airtable
import streamlit as st
from streamlit_chat import message
from datetime import datetime
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent

# Set environment variables
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Initialize Streamlit
st.image("socialai.jpg")

# Get current date and day of the week
current_date = datetime.today().strftime("%m/%d/%y")
day_of_week = datetime.today().weekday()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
current_day = days[day_of_week]

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = {}

if 'user_name' not in st.session_state:
    st.session_state.user_name = None

# Initialize Langchain components
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
memory_key = "history"
memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

# Define the tool for car dealership inventory retrieval (similar to your existing code)
file_1 = r'dealer_1_inventry.csv'
loader = CSVLoader(file_path=file_1)
docs_1 = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore_1 = FAISS.from_documents(docs_1, embeddings)
retriever_1 = vectorstore_1.as_retriever(search_type="similarity", search_kwargs={"k": 8})
tool1 = create_retriever_tool(
    retriever_1,
    "search_car_dealership_inventory",
    "Searches and returns documents regarding the car inventory and Input should be a single string strictly."
)

# Define the tool for business details retrieval (similar to your existing code)
business_details_text = [
    "working days: all Days except sunday",
    "working hours: 9 am to 7 pm",
    "Phone: (555) 123-4567",
    "Address: 567 Oak Avenue, Anytown, CA 98765, Email: jessica.smith@example.com",
    "dealer ship location: https://www.google.com/maps/place/Pine+Belt+Mazda/@40.0835762,-74.1764688,15.63z/data=!4m6!3m5!1s0x89c18327cdc07665:0x23c38c7d1f0c2940!8m2!3d40.0835242!4d-74.1742558!16s%2Fg%2F11hkd1hhhb?entry=ttu"
]
retriever_3 = FAISS.from_texts(business_details_text, OpenAIEmbeddings()).as_retriever()
tool3 = create_retriever_tool(
    retriever_3,
    "search_business_details",
    "Searches and returns documents related to business working days and hours, location and address details."
)

# Append all tools to the tools list
tools = [tool1, tool3]

# Initialize Airtable
airtable_api_key = st.secrets["AIRTABLE"]["AIRTABLE_API_KEY"]
os.environ["AIRTABLE_API_KEY"] = airtable_api_key
AIRTABLE_BASE_ID = "appAVFD4iKFkBm49q"
AIRTABLE_TABLE_NAME = "Question_Answer_Data"

# Streamlit UI setup
st.info("We're developing cutting-edge conversational AI solutions tailored for automotive retail, aiming to provide advanced products and support. As part of our progress, we're establishing an environment to check offerings and also check our website [engane.ai](https://funnelai.com/). This test application answers questions about Inventory, Business details, Financing, Discounts, and Offers. [Here](https://github.com/buravelliprasad/streamlit/blob/main/dealer_1_inventry.csv) is an inventory dataset you can explore and play with. The appointment dataset is [here](https://github.com/buravelliprasad/streamlit_dynamic_retrieval/blob/main/appointment.csv).")

# Initialize Langchain components
template = """
You're the Business Development Manager at our car dealership. When responding to inquiries, please adhere to the following guidelines:
Car Inventory Questions: If the customer's inquiry lacks specific details such as their preferred make, model, new or used car, and trade-in, kindly engage by asking for these specifics.
Specific Car Details: When addressing questions about a particular car, limit the information provided to make, year, model, and trim. For example, if asked about 'Do you have Jeep Cherokee Limited 4x4', the best answer should be 'Yes, we have Jeep Cherokee Limited 4x4: Year: 2022, Model: [Model], Make: [Make], Trim: [Trim]'.
Scheduling Appointments: If the customer's inquiry lacks specific details such as their preferred day, date, or time, kindly engage by asking for these specifics. Use today's date and day to find the appointment date from the user's input and check for appointment availability for that specific date and time. If the appointment schedule is not available, provide this link: [Appointment Link] to schedule an appointment by the user.
Encourage Dealership Visit: Our goal is to encourage customers to visit the dealership for test drives or receive product briefings from our team. After providing essential information on the car's make, model, color, and basic features, kindly invite the customer to schedule an appointment for a test drive or visit us for a comprehensive product overview by our experts.
Please maintain a courteous and respectful tone in your American English responses. If you're unsure of an answer, respond with 'I am sorry.' Make every effort to assist the customer promptly while keeping responses concise, not exceeding two sentences.
Feel free to use any tools available to look up for relevant information. Answer the question in not more than two sentences.
"""

details = f"Today's current date is {current_date} and today's week day is {current_day}."

input_template = template.format(details=details)
system_message = SystemMessage(content=input_template)
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

with container:
    if st.session_state.user_name is None:
        user_name = st.text_input("Your name:")
        if user_name:
            st.session_state.user_name = user_name

    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Type your question here (:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        result = agent_executor({"input": user_input})
        st.session_state.chat_history.append((user_input, result["output"]))
        session_key = f"{st.session_state.user_name}_{current_date}"
        if session_key not in st.session_state.chat_histories:
            st.session_state.chat_histories[session_key] = []
        st.session_state.chat_histories[session_key].append((user_input, result["output"]))

# Create a button to start a new chat session
new_chat_button = st.button("Start New Chat")

# When the new chat button is clicked, reset the chat history
if new_chat_button:
    st.session_state.chat_history = []
    user_input = ""  # Clear the input field

# Display Chat Histories in Sidebar
with st.sidebar:
    st.title("Chat History")
    session_key = f"{st.session_state.user_name}_{current_date}"
    chat_history = st.session_state.chat_histories.get(session_key, [])

    for i, (query, answer) in enumerate(chat_history):
        st.header(f"Query {i + 1}")
        st.subheader("User:")
        st.write(query)
        st.subheader("Assistant:")
        st.write(answer)

# Create a button to clear the current chat session history
if st.sidebar.button("Clear Current Chat History"):
    session_key = f"{st.session_state.user_name}_{current_date}"
    if session_key in st.session_state.chat_histories:
        st.session_state.chat_histories[session_key] = []

# Add a refresh button to clear the current chat session history and retrieve the previous chat history
if st.sidebar.button("Refresh Chat History"):
    session_key = f"{st.session_state.user_name}_{current_date}"
    if session_key in st.session_state.chat_histories:
        # Save the current chat history to a file with a name based on chat content
        current_chat = st.session_state.chat_history
        if current_chat:
            file_name = "_".join(current_chat[-1]).replace(" ", "_").replace("?", "").replace(":", "")
            file_name = f"{file_name}.txt"
            with open(file_name, "w") as file:
                for item in current_chat:
                    file.write(f"User: {item[0]}\nAssistant: {item[1]}\n\n")

        # Clear the current chat session history
        st.session_state.chat_history = []

        # Retrieve and display the previous chat history from Airtable
        try:
            airtable_api_key = st.secrets["AIRTABLE"]["AIRTABLE_API_KEY"]
            airtable = Airtable(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, api_key=airtable_api_key)

            # Retrieve previous chat history from Airtable based on the session_key
            previous_chat_history = airtable.get_all(formula=f"{{Session Key}} = '{session_key}'")

            # Display the previous chat history
            st.title("Previous Chat History")
            for record in previous_chat_history:
                user_input = record["fields"]["User Input"]
                assistant_response = record["fields"]["Assistant Response"]
                st.header("User:")
                st.write(user_input)
                st.header("Assistant:")
                st.write(assistant_response)

        except Exception as e:
            st.error(f"An error occurred while retrieving the previous chat history from Airtable: {e}")
