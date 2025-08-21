# import autosklearn.classification
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import streamlit as st

class MachineLearning():
    def __init__(self, llm, df, user_input = None):
        """
        Initialize the MachineLearning class.
        """
        h2o.init()
        self.llm = llm
        self.df = df
        self.hf = h2o.H2OFrame(df)
        print("============ H2O Frame ============")
        print(self.hf)
        self.train_df = None
        self.test_df = None
        self.user_input = user_input
        self.aml = None
        self.lb = None
        self.predictions = None
        self.target = None

    def data_split(self):
        """
        Split the data into training and testing sets.
        """
        self.train_df, self.test_df = self.hf.split_frame(ratios=[.8])
        print("============ Train Data ============")
        print(self.train_df)
        print("============ Test Data ============")
        print(self.test_df)

    def clean_up(self):
        """
        Clean up the environment.
        """
        self.train_df = None
        self.test_df = None
        self.aml = None
        self.lb = None
        self.predictions = None
        h2o.cluster().shutdown()

    def get_word_before_column(self, input_string):
        words = input_string.split()
        if 'column' in words:
            column_index = words.index('column')
            if column_index > 0:
                print("============ Target Column ============")
                print(words[column_index - 1])
                return words[column_index - 1]
        st.write("Column not found.")

    def process(self, user_input):
        """
        Process user input and return the assistant's response.
        """
        self.user_input = user_input
        if 'split' in self.user_input.lower().strip():
            self.data_split()
            st.write("Sample of Training Data: ")
            st.dataframe(self.train_df.as_data_frame().head())
            st.write("Sample of Testing Data: ")
            st.dataframe(self.test_df.as_data_frame().head())
            return "Data has been split into training and testing sets."

        if 'train' in self.user_input.lower().strip() and 'classification' in self.user_input.lower().strip():
            self.aml = H2OAutoML(max_runtime_secs=60)
            if self.train_df is None:
                return "You need to split the data first."
            else:
                self.target = self.get_word_before_column(self.user_input)
                self.train_df[self.target] = self.train_df[self.target].asfactor()
                self.aml.train(y=self.target, training_frame=self.train_df)
            return "Model has been trained."
        
        if 'train' in self.user_input.lower().strip() and 'regression' in self.user_input.lower().strip():
            self.aml = H2OAutoML(max_runtime_secs=60)
            if self.train_df is None:
                return "You need to split the data first."
            else:
                self.target = self.get_word_before_column(self.user_input)
                self.aml.train(y=self.target, training_frame=self.train_df)
            return "Model has been trained."

        if 'leaderboard' in self.user_input.lower().strip():
            if self.aml is None:
                return "You need to train the model first."
            else:
                self.lb = self.aml.leaderboard
                st.write(self.lb)
            return "Leaderboard has been displayed."
            
        if 'predict' in self.user_input.lower().strip():
            if self.aml is None:
                return "You need to train the model first."
            else:
                self.predictions = self.aml.leader.predict(self.test_df)
                st.dataframe(self.predictions.as_data_frame().head())
            return "Predictions have been made."
        
        if 'performance' in self.user_input.lower().strip():
            if self.aml is None:
                return "You need to train the model first."
            else:
                st.write(self.aml.leader.model_performance(self.test_df))
            return "Model performance has been displayed."
        
        if 'clear' in self.user_input.lower().strip():
            clear_statement = self.clean_up()
            return "All data has been cleared."
        
        return "Machine learning is done. Can I help you with anything else?"