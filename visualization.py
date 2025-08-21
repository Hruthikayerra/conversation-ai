import re
import langchain_experimental
from langchain_experimental.agents import create_pandas_dataframe_agent
import streamlit as st
import pandas as pd
#from pandasai import SmartDataframe
import json

class Visualization:
    def __init__(self, llm, df, user_input):
        self.df = df
        self.llm = llm
        self.user_input = user_input
    
    def ask_agent(self, agent, user_input: str, df: pd.DataFrame):
        """
        Query an agent and return the response as a string.
        """
        # Prepare the prompt with query guidelines and formatting
        viz_prompt = (
            """
            You are an expert in data visualization. You have been asked to create a visualization for the following data.
            Here is the sample of data you need to visualize which is stored in df variable:
            """
            + df.head().to_string()
            +
            f"""Here is the user question: {self.user_input}"""
            +
            """- Do not load the data again. Use the df variable to visualize the data.
               - Give proper title and labels to the chart.
               - Important: Display using streamlit st.
               - Write the code in the code block."""
        )

        # Run the prompt through the agent and capture the response.
        response = agent.run(viz_prompt)

        return response
    
    def write_response(self, response_dict: dict):
        """
        Write a response from an agent to a Streamlit app.

        Args:
            response_dict: The response from the agent.

        Returns:
            None.
        """

        # Check if the response is an answer.
        if "answer" in response_dict:
            st.write(response_dict["answer"])

        # Check if the response is a bar chart.
        if "bar" in response_dict:
            data = response_dict["bar"]
            df = pd.DataFrame(data)
            df.set_index("columns", inplace=True)
            st.bar_chart(df)

        # Check if the response is a line chart.
        if "line" in response_dict:
            data = response_dict["line"]
            df = pd.DataFrame(data)
            df.set_index("columns", inplace=True)
            st.line_chart(df)

        # Check if the response is a table.
        if "table" in response_dict:
            data = response_dict["table"]
            df = pd.DataFrame(data["data"], columns=data["columns"])
            st.table(df)

    def decode_response(self, response: str) -> dict:
        """This function converts the string response from the model to a dictionary object.
        Args:
            response (str): response from the model
        Returns:
            dict: dictionary with response data
        """
        return json.loads(response)
    
    def process(self):
        """
        Process user input and return the assistant's response.
        """
        df = self.df

        # to disable the warning
        st.set_option('deprecation.showPyplotGlobalUse', False)

        agent = create_pandas_dataframe_agent(self.llm, 
                                              self.df, 
                                              verbose=True, 
                                              agent_type="openai-tools",
                                              agent_executor_kwargs={"handle_parsing_errors": True},
                                              )
        
        response = self.ask_agent(agent,
                                  self.user_input,
                                  df= df)
        pattern = r'`python\n(.*?)\n`'
        code_snippet = re.findall(pattern, response, re.DOTALL)
        code = '\n'.join(code_snippet)
        # Execute the modified code
        exec(code)
        # response = self.ask_agent(agent = agent,
        #                           query = self.user_input)
        #exec(response)
        print("======== Response =========")
        print(response)
        # decoded_response = self.decode_response(response)
        # self.write_response(decoded_response)
        #return fig
