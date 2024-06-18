import uuid
from langserve import RemoteRunnable

conversation_id = str(uuid.uuid4())

remote_chain = RemoteRunnable("http://localhost:8000/", cookies={"user_id": "test_user"})

while True:
    usr_input = input("User: ")
    print(remote_chain.invoke({"input": usr_input},
    {"configurable": {"conversation_id": conversation_id}}))