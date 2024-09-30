from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads

from operator import itemgetter

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

def get_response(llm_model, retriever, user_input, chat_history=[]):
    
    template_prompt_generate_queries = """
    You are an AI assistant designed to improve search results by generating diverse versions of a user's query for document retrieval. 
    Your task is to rephrase and expand on the user’s original question in five different ways, focusing on different aspects of the query while considering the conversation history.
    This will help to capture a broader range of relevant documents from the vector database.

    Be sure to:
    1. Reframe the question from multiple angles (e.g., broaden, narrow, or shift focus).
    2. Use synonyms, alternative phrasing, or different assumptions to explore variations.
    3. Consider potential clarifications or ambiguities in the original question.
    4. Incorporate information from the chat history where relevant.

    Separate each alternative question with a newline.

    Original question: {question}

    Chat history for context: {chat_history}
    """

    prompt_generate_queries = ChatPromptTemplate.from_template(template_prompt_generate_queries)

    gen_queries_chain = (
        prompt_generate_queries
        | llm_model
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    retrieval_chain = gen_queries_chain | retriever.map() | get_unique_union
    
    prompt_template = """Answer the following question based on this context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    final_rag_chain = (
        {"context": retrieval_chain, 
        "question": itemgetter("question")} 
        | prompt
        | llm_model
        | StrOutputParser()
    )
    response = final_rag_chain.invoke({
        "question":user_input,
        "chat_history":chat_history
    })
    return response