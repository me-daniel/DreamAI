from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from dotenv import load_dotenv

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

ephemeral_chat_history_for_chain = ChatMessageHistory()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "Act as a therapist. You are an expert in imagery rehearsal therapy and guide the user step by step through the method by engaging them in a socratic dialogue.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser =StrOutputParser()

chain = prompt | llm | output_parser

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: ephemeral_chat_history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
)
