import os
import pandas as pd
import streamlit as st
from streamlit_chat import message
from langchain.sql_database import SQLDatabase
from langchain.chains import (
    SQLDatabaseChain,
    SQLDatabaseSequentialChain
)
from langchain.chat_models import ChatOpenAI
from sqlalchemy import create_engine

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'history' not in st.session_state:
    st.session_state['history'] = []


def run():
    st.set_page_config(
        page_title="LangChain",
        page_icon=":robot:"
    )
    st.title("LangChain Chatbot")

    db_user = 'postgres'
    db_password = 'jiaming1234'
    db_host = 'localhost:5432'
    db_name = 'edm_my'  # 'jming_DB'
    db = SQLDatabase.from_uri(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}')

    chat_placeholder = st.container()
    input_placeholder = st.container()

    user_input = input_placeholder.text_input(
        label='Write something',
        value=''
    )

    if user_input != '':
        chain = SQLDatabaseChain.from_llm(
            llm=ChatOpenAI(
                openai_api_key=st.secrets['OPENAI_API_KEY'],
                model_name='gpt-3.5-turbo',  # gpt-3.5-turbo, davinci, text-davinci-003
                temperature=0.0,
            ),
            db=db,
            verbose=True,
            return_intermediate_steps=True,
        )

        result = chain(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(result['result'])
        query = result['intermediate_steps'][2]['sql_cmd']
        st.code(query, language='sql')

        if query is not None:
            conn_string = f'postgresql://{db_user}:{db_password}@{db_host}/{db_name}'
            db = create_engine(conn_string)
            conn = db.connect()

            df = pd.read_sql(query, conn)
            st.dataframe(
                data=df,
                use_container_width=True,
                hide_index=True
            )
            encoded_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='Download data as .csv',
                data=encoded_data,
                file_name='results.csv',
                mime='text/csv'
            )

    if st.session_state["generated"]:
        with chat_placeholder:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["generated"][i], key=str(i))


if __name__ == "__main__":
    run()
