import os
import streamlit as st
from streamlit_chat import message
import pandas as pd
import openai
from langchain.sql_database import SQLDatabase
from langchain.chains import (
    LLMChain,
    ConversationChain,
    SimpleSequentialChain,
    ConversationalRetrievalChain,
    SQLDatabaseChain,
    SQLDatabaseSequentialChain
)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# os.environ['OPENAI_API_TOKEN'] = st.secrets['OPENAI_API_KEY']
# openai.api_key = st.secrets['OPENAI_API_KEY']

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
    db_name = 'jming_DB'
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
            # use_query_checker=True,
            return_intermediate_steps=True,
        )

        # result = chain(
        #     {
        #         "question": user_input,
        #         "chat_history": st.session_state['history']
        #     }
        # )
        # st.session_state['history'].append((user_input, result))

        result = chain(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(result['result'])
        st.code(result['intermediate_steps'][2]['sql_cmd'], language='sql')

    if st.session_state["generated"]:
        with chat_placeholder:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["generated"][i], key=str(i))


if __name__ == "__main__":
    run()
