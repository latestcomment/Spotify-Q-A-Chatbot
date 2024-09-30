import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage

from dotenv import load_dotenv

from html_templates import custom_css, user_template, bot_template
from generate import get_response
from store_vector import get_vectorstore

load_dotenv()
    
def main():
    st.set_page_config(page_title="Spotify Reviews Q&A Chatbot",
                       page_icon=":nerd_face:")    
    st.write(custom_css, unsafe_allow_html=True)

    st.header("Discover Spotify Reviews :bookmark_tabs:")
    
    # init variables
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = ChatOpenAI(model_name="gpt-4o", temperature=0)
    if "embedding" not in st.session_state:
        st.session_state.embedding = OpenAIEmbeddings()
    if 'vectorstore' not in st.session_state:
        with st.spinner('Initializing knowledge base ...'):
            st.session_state.vectorstore = get_vectorstore(st.session_state.embedding)
    if "retriever" not in st.session_state:
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # user input
    user_input = st.chat_input("Type your message here...")
    if user_input is not None and user_input != "":
        response = get_response(st.session_state.llm_model,
                                st.session_state.retriever,
                                user_input,
                                st.session_state.chat_history)

        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

if __name__=="__main__":
    main()