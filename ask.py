from langchain.prompts.chat import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from typing import Optional
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import pandas as pd
from visualization import Visualization
from machine_learning import MachineLearning
import streamlit as st
class Ask:
    def __init__(self):
        # Define the language model
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=1000,
            #openai_api_key="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",
            openai_api_key="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",
            )
    
        memory = ConversationBufferMemory(memory_size=3)
        self.conversation = ConversationChain(llm=self.llm, 
                                              verbose=True, 
                                              memory=memory,
                                              )
        self.i = 0

    def process(self, user_input, db= None):
        """
        Process user input and return the assistant's response.
        """

        tools = ["general", "sql", "machine-learning", "visualization"]
        
        prompt = ChatPromptTemplate.from_template("""
        You are helpful assistant that helps pick a tool based on the user question.
        Here are the tools you can use: {tools}.
        Use 'sql' tool when the user question is related to data.
        Use 'visualization' tool when the user question is related to data visualizations like charts, graphs, etc.
        Here is the user question: {input}
        Your response should be tool name.  
        """)
        chain = prompt | self.llm
        decision = chain.invoke({'input': user_input,
                                 'tools': ', '.join(tools)})
        if decision.content == "sql":
            if db is None:
                return "I can't help you with SQL because the database is not connected."
            else:
                # Define the SQL agent
                sql_agent = create_sql_agent(
                    llm=self.conversation.llm,
                    db= db,
                    agent_type="openai-tools", 
                    verbose=True
                )
                return sql_agent.invoke(user_input)['output']
        elif decision.content == "visualization" or decision.content == "machine-learning":
            """
            Logic to handle visualization and machine learning
            """
            # prompt to create sql query to get data for visualization
            query_prompt = ChatPromptTemplate.from_template("""
            You are an SQL expert that provides a SQL query to get all the data for table mentioned in user’s question.
            Here are the available tables in database: {tables}.
            Here is the user’s question: {input}
            Your response should be 'SELECT * FROM table_name;'
            relating table_name to the user question based on available tables.
            Do not provide explanation.
            """)
            chain = query_prompt | self.llm
            query = chain.invoke({'tables': db.run("SHOW TABLES;", fetch="all"),
                                  'input': user_input})
            
            # printing table names for debugging
            print("============ Table names ============")
            print(db.run("SHOW TABLES;", fetch="all"))

            query_result = db.run(query.content, fetch="cursor")
            df = pd.DataFrame(list(query_result.mappings()))
            
            if decision.content == "visualization":
                st.session_state.visualization_instance = Visualization(self.llm, df, user_input)
                print("============ Visualization Instance Inititated ============")
                st.session_state.visualization_instance.process()
                return "Visualization is done. Can I help you with anything else?"
            
            elif decision.content == "machine-learning":
                if 'machine_learning_instance' not in st.session_state:
                    st.session_state.machine_learning_instance = MachineLearning(self.llm, df)
                    print("============ Machine Learning Instance Inititated ============")
                ml_return = st.session_state.machine_learning_instance.process(user_input)
                return ml_return
            
        else:
            """
            Logic to handle general questions
            """
            template_string = """You are a Data Science Expert capable of assisting users with their 
            database-related queries, creating visualizations, performing machine learning tasks like modeling, 
            and answering general questions. {input}"""
            
            general_response  = self.conversation.invoke(template_string.format(input=user_input))
            return general_response['response']