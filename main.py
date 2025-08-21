import streamlit as st
import os
from connection import DatabaseConnection
from ask import Ask
from langchain_community.utilities import SQLDatabase
import pandas as pd

# Title and header
st.title("Conversational AI")
st.text("Data Science and insights at your fingertips!")

# Database configuration section
st.sidebar.header("Database Configuration")
db_type = st.sidebar.selectbox("Database Type", ("mysql", "sqlite"))
host = st.sidebar.text_input("Host")
user_name = st.sidebar.text_input("User Name")
password = st.sidebar.text_input("Password", type="password")
db_name = st.sidebar.text_input("Database Name")

uri = None 

# Connect to the database
if st.sidebar.button("Connect"):
    db = DatabaseConnection(db_type, user_name, db_name, host, password)
    st.session_state.conn, st.session_state.uri = db.get_connection()
    if st.session_state.conn:
        st.success("Connected successfully!")
        st.balloons()
    else:
        st.error("Failed to connect.")

# Initialize llm_output
llm_output = None

# Ask instance
if "ask_instance" not in st.session_state:
    st.session_state.ask_instance = Ask()
    print("============ ASK Intance Inititated ============")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt :=st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    #uri = "mysql+mysqlconnector://root:12345678@localhost/finalpro"
    llm_output = st.session_state.ask_instance.process(prompt, SQLDatabase.from_uri(st.session_state.uri))
    # Display assistant message in chat message container
    with st.chat_message("assistant"):
        st.markdown(llm_output)

    st.session_state.messages.append({"role": "assistant", "content": llm_output})

