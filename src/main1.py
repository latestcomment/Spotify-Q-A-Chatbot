import pandas as pd
import numpy as np
import streamlit as st

from langchain.text_splitter import CharacterTextSplitter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, create_retrieval_chain

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain_chroma import Chroma
# import chromadb

import os
from dotenv import load_dotenv

from html_templates import chat_css, user_template, bot_template

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # vector store
# chroma_client = chromadb.Client()

# raw_data to dataframe
data_df = pd.read_csv('data/SPOTIFY_REVIEWS.csv')

def split_text(text):
    splitter = CharacterTextSplitter(separator=" ",
                                     chunk_size=50,
                                     chunk_overlap=10,
                                     length_function=len)

    chunks = splitter.split_text(text)
    return chunks

def get_docs(dataframe):
    docs = []
    i = 0
    while i < 100: # use dataframe.shape[0] for full docs
        text = dataframe['review_text'][i]
        chunks = split_text(text)
        for chunk in chunks:
            docs.append(chunk)
        i += 1
    return docs

def get_vectorstore(embedding, texts):
    vectorstore = Chroma(embedding_function=embedding,
                         collection_name="spotify-test",
                         persist_directory="./chroma_db")
    
    # indexing
    vectorstore.add_texts(texts)
    return vectorstore

def get_chain(vectorstore):

    llm_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    prompt_template = """
    Given the following context from retrieved documents: {context}
    and the previous conversation history: {chat_history},
    
    Answer the question: {input}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | llm_model

    retriever = vectorstore.as_retriever()
    retriever_prompt_template = """
    Based on the conversation so far: {chat_history},
    generate a search query that will retrieve relevant information for the question.

    Question: {input}
    """
    retriever_prompt = ChatPromptTemplate.from_template(retriever_prompt_template)

    # Use a history-aware retriever that integrates the chat history
    history_aware_retriever = create_history_aware_retriever(
        llm=llm_model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    return retrieval_chain

def handle_user_input(chain, user_query):
    response = chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input" : user_query
        })
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response['answer'].content))

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    
def main():
    texts = get_docs(data_df)
    embedding = OpenAIEmbeddings()
    vectorstore = get_vectorstore(embedding, texts)
    chain = get_chain(vectorstore)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    st.set_page_config(page_title="Spotify Reviews Q&A Chatbot",
                       page_icon=":nerd_face:")
    
    st.write(chat_css, unsafe_allow_html=True)
    st.header("Discover Spotify Reviews :bookmark_tabs:")

    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        handle_user_input(chain, user_query)

if __name__=="__main__":
    main()
        