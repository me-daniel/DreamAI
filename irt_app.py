from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

from prompt_templates import router_template, recording_template, rewriting_template, summary_template, summary_template_chat

import os
from dotenv import load_dotenv

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

ephemeral_chat_history_for_chain = ChatMessageHistory()
ephemeral_chat_history_2 = ChatMessageHistory()
transcript = ""
openAI = ChatOpenAI(model="gpt-3.5-turbo")
groq = ChatGroq(temperature=0, model_name="llama3-8b-8192")
output_parser =StrOutputParser()


# Define the chains 
router_chain = router_template | groq | StrOutputParser()

recording_chain = recording_template | openAI | StrOutputParser()

rewriting_chain =  rewriting_template | openAI | StrOutputParser()

summary_chain = summary_template | openAI | StrOutputParser()

summary_chain_chat = summary_template_chat | openAI | StrOutputParser()

# Routing function 
def route(info):
    if "recording" in info["stage"].lower():
        return recording_chain
    elif "rewriting" in info["stage"].lower():
        return rewriting_chain
    else:
        return summary_chain_chat

# Wrap the router chain with message history
router_chain_chat_history = RunnableWithMessageHistory(
    router_chain,
    lambda session_id: ephemeral_chat_history_2,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Wrap the subchains with message history 
router_with_history = RunnableWithMessageHistory(
    RunnableLambda(route),
    lambda session_id: ephemeral_chat_history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
)
    
#full_chain = {"stage": router_chain_chat, "input": lambda x: x["input"], "transcript": lambda x : x["transcript"]} | router_with_history

full_chain = {"stage": router_chain_chat_history, "input": lambda x: x["input"]} | router_with_history

# Chatbot loop
while False:
    user_in = input("User: ")
    response = full_chain.invoke({"input": user_in},
                                 {"configurable": {"session_id": 'unused'}})
    print(response)
 