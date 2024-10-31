# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 08:21:31 2024

@author: Rehan ali
"""



# Imports
import streamlit as st
import snowflake.connector
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import io
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Set seaborn style
sns.set_theme(style='darkgrid')

# Constants
OPENROUTER_API_KEY = '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'

# Schema Definition
SNOWFLAKE_SCHEMA = """
{
  "table": "query_history",
  "columns": {
    "QUERY_ID": "string",
    "QUERY_TEXT": "string",
    "DATABASE_NAME": "string",
    "SCHEMA_NAME": "string",
    "QUERY_TYPE": "string",
    "SESSION_ID": "number",
    "USER_NAME": "string",
    "ROLE_NAME": "string",
    "WAREHOUSE_NAME": "string",
    "WAREHOUSE_SIZE": "string",
    "WAREHOUSE_TYPE": "string",
    "CLUSTER_NUMBER": "number",
    "QUERY_TAG": "string",
    "EXECUTION_STATUS": "string",
    "ERROR_CODE": "number",
    "ERROR_MESSAGE": "string",
    "START_TIME": "string",
    "END_TIME": "string",
    "TOTAL_ELAPSED_TIME": "number",
    "BYTES_SCANNED": "number",
    "ROWS_PRODUCED": "number",
    "COMPILATION_TIME": "number",
    "EXECUTION_TIME": "number",
    "CREDITS_USED_CLOUD_SERVICES": "number"
  }
}
{
  "table": "warehouse_metering_history",
  "columns": {
    "START_TIME": "string",
    "END_TIME": "string",
    "WAREHOUSE_NAME": "string",
    "CREDITS_USED": "number",
    "CREDITS_USED_COMPUTE": "number",
    "CREDITS_USED_CLOUD_SERVICES": "number"
  }
}
{
  "table": "query_attribution_history",
  "columns": {
    "QUERY_ID": "string",
    "WAREHOUSE_NAME": "string",
    "USER_NAME": "string",
    "START_TIME": "string",
    "END_TIME": "string",
    "CREDITS_ATTRIBUTED_COMPUTE": "number"
  }
}
"""


# System Prompts
SQL_GENERATION_PROMPT = """You are a specialized SQL assistant for Snowflake cost analysis. Your role is to convert natural language questions into precise SQL queries using the provided schema. Follow these strict guidelines:

1. Schema Rules:
   - Only use tables and columns defined in the schema
   - Respect data types specified in the schema
   - Use proper joins based on logical key relationships
   - Always use uppercase for column names

2. Query Structure:
   - Write clear, optimized SQL queries
   - Use appropriate aggregations and groupings
   - Include proper date/time handling for temporal analysis
   - Add appropriate filters and conditions

3. Cost Analysis Focus:
   - Prioritize credit usage and cost-related metrics
   - Consider performance metrics when relevant
   - Include warehouse utilization metrics when appropriate
   - Handle time-based aggregations properly

4. Output Format:
   - Return ONLY the SQL query, no explanations
   - Ensure queries start with SELECT
   - Use proper indentation and formatting
   - Include appropriate ORDER BY clauses
   - Limit results when appropriate

Schema: {schema}

Generate SQL for the following question: """

SQL_EXPLANATION_PROMPT = """As a Snowflake SQL expert, explain the following SQL query in a clear, detailed manner. Break down the explanation into sections:

1. Overview:
   - What is the main purpose of the query
   - What kind of information it retrieves

2. Query Structure:
   - Tables used and their relationships
   - Key columns being selected
   - Filtering conditions
   - Aggregations and groupings

3. Business Context:
   - How the results can be interpreted
   - What insights can be derived
   - Relevant business metrics

Query to explain:
{query}

Schema:
{schema}

Provide a clear, concise explanation that would help a business analyst understand both the technical and business aspects of this query."""

SQL_ERROR_CORRECTION_PROMPT = """You are an expert SQL debugger specializing in Snowflake queries. Your task is to analyze and fix SQL queries that have resulted in errors. Follow these guidelines:

1. Error Analysis:
   - Examine the error message carefully
   - Identify syntax or logical issues
   - Check schema compliance
   - Verify column names and types

2. Fix Requirements:
   - Maintain the original query intent
   - Use only schema-defined tables/columns
   - Follow Snowflake best practices
   - Ensure proper syntax

Original Query:
{query}

Error Message:
{error}

Schema:
{schema}

Return only the corrected SQL query:"""


def create_visualization(df, question):
    try:
        if df is None or df.empty:
            return None

        prompt = f"""Create a Matplotlib visualization for:
Table Data:
{df.head(10).to_string()}

Question: {question}

Return ONLY Python code that:
1. Uses matplotlib/seaborn
2. Uses 'df' DataFrame
3. Sets style and color (#0066FF)
4. Has clear title and labels
5. Uses plt.figure(figsize=(10, 6))
6. Ends with st.pyplot(plt)
7. Clears figure with plt.close()"""

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response.status_code == 200:
            viz_code = response.json()['choices'][0]['message']['content'].strip()
            if "```" in viz_code:
                viz_code = viz_code.split("```")[1]
                if viz_code.startswith("python"):
                    viz_code = viz_code[6:]
            
            viz_code = viz_code.strip()
            
            st.markdown('<div class="chart-area">', unsafe_allow_html=True)
            
            exec_globals = {
                'df': df,
                'plt': plt,
                'sns': sns,
                'st': st,
                'pd': pd,
                'np': np
            }
            
            exec(viz_code, exec_globals)
            st.markdown('</div>', unsafe_allow_html=True)
            return True

    except Exception as e:
        print(f"Error in create_visualization: {str(e)}")
        st.error("Failed to create visualization")
        return None

def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'query_history' not in st.session_state:
        st.session_state['query_history'] = []
    if 'current_query' not in st.session_state:
        st.session_state['current_query'] = None
    if 'current_results' not in st.session_state:
        st.session_state['current_results'] = None
    if 'current_explanation' not in st.session_state:
        st.session_state['current_explanation'] = None
    if 'current_question' not in st.session_state:
        st.session_state['current_question'] = None

def clear_session():
    for key in st.session_state.keys():
        del st.session_state[key]

def update_session_with_results(question, query, df, explanation):
    st.session_state['current_query'] = query
    st.session_state['current_results'] = df
    st.session_state['current_explanation'] = explanation
    st.session_state['current_question'] = question
    st.session_state['query_history'].append({
        'question': question,
        'sql_query': query,
        'timestamp': datetime.now()
    })
    
    

def create_snowflake_connection(user, password, account):
    try:
        conn = snowflake.connector.connect(
            user=user,
            password=password,
            account=account,
            database='SNOWFLAKE',
            schema='ACCOUNT_USAGE',
            client_session_keep_alive=True
        )
        return conn
    except Exception as e:
        raise Exception(f"Failed to connect to Snowflake: {str(e)}")

def convert_query_to_sql(question):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [
                    {
                        "role": "system", 
                        "content": SQL_GENERATION_PROMPT.format(schema=SNOWFLAKE_SCHEMA)
                    },
                    {
                        "role": "user", 
                        "content": question
                    }
                ]
            }
        )
        
        if response.status_code == 200:
            sql_query = response.json()['choices'][0]['message']['content'].strip()
            return sql_query if sql_query.upper().startswith('SELECT') else None
        return None
    except Exception as e:
        print(f"Error in convert_query_to_sql: {str(e)}")
        return None

def execute_query_with_retry(conn, query, max_retries=2):
    retries = 0
    current_query = query
    
    while retries <= max_retries:
        try:
            cursor = conn.cursor()
            cursor.execute(current_query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            cursor.close()
            return pd.DataFrame(results, columns=columns), current_query
        except Exception as e:
            if retries < max_retries:
                corrected_query = fix_sql_query(current_query, str(e))
                if corrected_query:
                    current_query = corrected_query
                    retries += 1
                    continue
            raise Exception("Could not execute query")

def fix_sql_query(original_query, error_message):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [
                    {
                        "role": "system", 
                        "content": SQL_ERROR_CORRECTION_PROMPT.format(
                            query=original_query,
                            error=error_message,
                            schema=SNOWFLAKE_SCHEMA
                        )
                    }
                ]
            }
        )
        
        if response.status_code == 200:
            corrected_query = response.json()['choices'][0]['message']['content'].strip()
            return corrected_query if corrected_query.upper().startswith('SELECT') else None
        return None
    except Exception as e:
        print(f"Error in fix_sql_query: {str(e)}")
        return None

def get_query_explanation(query):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [
                    {
                        "role": "system", 
                        "content": SQL_EXPLANATION_PROMPT.format(
                            query=query,
                            schema=SNOWFLAKE_SCHEMA
                        )
                    }
                ]
            }
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        return None
    except Exception as e:
        print(f"Error in get_query_explanation: {str(e)}")
        return None
    
def get_base64_logo():
    try:
        with open("E:/Noodle Analyst/noodleseedlogo.png", "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        print(f"Error loading logo: {str(e)}")
        return ""

def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * { font-family: 'Inter', sans-serif; }
        
        .stApp { background-color: #f8fafc; }

        /* Hide sidebar, header, and footer */
        header[data-testid="stHeader"],
        section[data-testid="stSidebar"],
        footer { display: none; }

        /* Query Interface Styles */
        .query-header-container {
            display: flex;
            align-items: center;
            padding: 1.5rem;
            background-color: white;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 2rem;
        }

        .query-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-left: 1rem;
            color: #0066FF;
        }

        .query-form-container {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid #e2e8f0;
            margin-bottom: 1rem;
        }

        /* Input Styles */
        div[data-testid="stTextInput"] input,
        div[data-testid="stPasswordInput"] input {
            background-color: white !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            padding: 0.75rem !important;
            font-size: 0.875rem !important;
            color: #1e293b !important;
        }

        /* Textarea for Query */
        .query-textarea textarea {
            background-color: white !important;
            color: black !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            padding: 0.75rem !important;
            font-size: 0.875rem !important;
            min-height: 100px !important;
        }

        /* Button Styles */
        .stButton button, 
        div[data-testid="stForm"] button[kind="primaryFormButton"] {
            background-color: #0066FF !important;
            color: white !important;
            border: none !important;
            padding: 0.75rem 1.5rem !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            border-radius: 8px !important;
            width: 100% !important;
            box-shadow: 0 1px 3px rgba(0, 102, 255, 0.1) !important;
            transition: all 0.2s ease !important;
        }

        .stButton button:hover,
        div[data-testid="stForm"] button[kind="primaryFormButton"]:hover {
            background-color: #0052CC !important;
            box-shadow: 0 2px 4px rgba(0, 102, 255, 0.2) !important;
        }

        /* Tabs */
        .stTabs {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid #e2e8f0;
        }

        div[data-testid="stHorizontalBlock"] button[data-baseweb="tab"] {
            background-color: transparent !important;
            color: black !important;
            font-weight: 500 !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            margin-right: 0.5rem !important;
        }
        
        div[data-testid="stHorizontalBlock"] button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #0066FF !important;
            color: white !important;
        }

        /* Explanation text */
        .explanation-text {
            color: black !important;
        }
        </style>
    """, unsafe_allow_html=True)

def login_form():
    st.markdown("""
        <div class="login-header-container">
            <img src="data:image/png;base64,{}" class="noodle-logo">
            <h1 class="login-title">Noodle Analyst</h1>
        </div>
    """.format(get_base64_logo()), unsafe_allow_html=True)
    
    with st.form("login_form", clear_on_submit=False):
        with st.container():
            st.text_input("Username", key="username", placeholder="Enter your Snowflake username")
            st.text_input("Password", type="password", key="password", placeholder="Enter your Snowflake password")
            st.text_input("Account", key="account", placeholder="e.g., xy12345.us-east-1")
        submitted = st.form_submit_button("Connect")
        
        if submitted:
            try:
                conn = create_snowflake_connection(
                    st.session_state.username,
                    st.session_state.password,
                    st.session_state.account
                )
                st.session_state['conn'] = conn
                st.session_state['logged_in'] = True
                st.rerun()
            except Exception as e:
                st.error("Connection failed. Please check your credentials.")
                
                

def query_interface():
    st.markdown("""
        <div class="query-header-container">
            <img src="data:image/png;base64,{}" class="noodle-logo">
            <h1 class="query-title">Noodle Analyst</h1>
        </div>
    """.format(get_base64_logo()), unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.markdown('<div class="query-form-container">', unsafe_allow_html=True)
        with st.form("query_form"):
            question = st.text_area("Enter your question:", key="query_textarea", placeholder="Type your question here...", height=100)
            execute_button = st.form_submit_button("Execute")
            
            if execute_button and question:
                with st.spinner('Processing...'):
                    sql_query = convert_query_to_sql(question)
                    if sql_query:
                        try:
                            explanation = get_query_explanation(sql_query)
                            df, final_query = execute_query_with_retry(st.session_state['conn'], sql_query)
                            update_session_with_results(question, final_query, df, explanation)
                            st.rerun()
                        except Exception as e:
                            st.warning("Unable to process question. Please rephrase.")
                    else:
                        st.warning("Unable to understand question. Please rephrase.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state['query_history']:
            st.markdown("### Previous Queries")
            for entry in reversed(st.session_state['query_history']):
                with st.expander(f"Q: {entry['question']}", expanded=False):
                    st.code(entry['sql_query'], language='sql')
                    st.markdown(f"Asked at: {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Clear History"):
                st.session_state['query_history'] = []
                st.session_state['current_query'] = None
                st.session_state['current_results'] = None
                st.session_state['current_explanation'] = None
                st.rerun()
        with col2:
            if st.button("Logout"):
                if 'conn' in st.session_state:
                    st.session_state['conn'].close()
                clear_session()
                st.rerun()

    with right_col:
        query_tab, results_tab, explanation_tab = st.tabs(["Generated Query", "Results & Visualization", "Query Explanation"])

        with query_tab:
            if st.session_state.get('current_query'):
                st.code(st.session_state['current_query'], language='sql')

        with results_tab:
            if st.session_state.get('current_results') is not None:
                csv = st.session_state['current_results'].to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download CSV", csv, "snowflake_data.csv", "text/csv")
                st.dataframe(st.session_state['current_results'])
                create_visualization(st.session_state['current_results'], st.session_state.get('current_question', ''))

        with explanation_tab:
            if st.session_state.get('current_explanation'):
                st.markdown(f'<div class="explanation-text">{st.session_state["current_explanation"]}</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Noodle Data Analyst",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    apply_custom_css()
    init_session_state()
    
    if not st.session_state.get('logged_in'):
        login_form()
    else:
        query_interface()

if __name__ == "__main__":
    main()