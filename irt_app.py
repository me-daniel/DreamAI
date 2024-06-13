from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, Runnable

from prompt_templates import router_template, recording_template, rewriting_template, summary_template

import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

ephemeral_chat_history_for_chain = ChatMessageHistory()
ephemeral_chat_history_2 = ChatMessageHistory()
transcript = ""
openAI = ChatOpenAI(model="gpt-4o")
groq = ChatGroq(temperature=0, model_name="llama3-8b-8192")
llama3 = ChatGroq(temperature=0, model_name="llama3-70b-8192")
output_parser =StrOutputParser()

# Define the chains 
router_chain = router_template | groq | StrOutputParser()

recording_chain = recording_template | openAI | StrOutputParser()

rewriting_chain =  rewriting_template | openAI | StrOutputParser()

summary_chain = summary_template | openAI | StrOutputParser()


# Routing function 
def route(info):
    if "recording" in info["stage"].lower():
        return recording_chain
    elif "rewriting" in info["stage"].lower():
        return rewriting_chain
    else:
        return summary_chain
    
def clean_history(chat_history):
    if len(chat_history.messages) > 1:
        chat_history.messages = chat_history.messages[:-2]
    return chat_history

class PostProcessingRunnable(Runnable):
    def __init__(self, base_chain, post_processing_fn):
        self.base_chain = base_chain
        self.post_processing_fn = post_processing_fn

    def invoke(self, inputs, config):
        result = self.base_chain.invoke(inputs, config)
        chat_history = ephemeral_chat_history_for_chain
        self.post_processing_fn(chat_history)
        return result

# Wrap the router chain with message history
router_chain_chat_history = RunnableWithMessageHistory(
    router_chain,
    #lambda session_id: ephemeral_chat_history_for_chain,
    lambda session_id: ephemeral_chat_history_for_chain,
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
  
# Create the PostProcessingRunnable
post_processing_runnable = PostProcessingRunnable(router_chain_chat_history, clean_history)

# Create the full chain configuration
full_chain = {"stage": post_processing_runnable, "input": lambda x: x["input"]} | router_with_history

# Chatbot loop
while True:
    user_in = input("User: ")
    response = full_chain.invoke({"input": user_in},
                                 {"configurable": {"session_id": 'unused'}})
    print("-----------------------------------------")
    print("AI: " + response)
    print("-----------------------------------------")
    
 
