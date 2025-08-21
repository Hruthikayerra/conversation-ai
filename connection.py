import streamlit as st
import mysql.connector
import sqlite3

class DatabaseConnection:
    def __init__(self, db_type, user, db_name, host, password):
        self.db_type = db_type
        self.user = user
        self.db_name = db_name
        self.host = host
        self.password = password

    def get_connection(self):
        if self.db_type == "mysql":
            try:
                conn = mysql.connector.connect(
                    user=self.user,
                    password=self.password,
                    host=self.host,
                    database=self.db_name
                )
                uri = f"mysql+mysqlconnector://{self.user}:{self.password}@{self.host}/{self.db_name}"
                return conn, uri
            except Exception as e:
                st.error(f"Connection failed: {e}")
                return None, None
        elif self.db_type == "sqlite":
            try:
                return sqlite3.connect(self.host), None # Assuming host is the file path
            except Exception as e:
                st.error(f"Connection failed: {e}")
                return None, None