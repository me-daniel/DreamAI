import uuid
from langserve import RemoteRunnable

conversation_id = str(uuid.uuid4())

remote_chain = RemoteRunnable("http://localhost:8000/", cookies={"user_id": "test_user"})

while True:
    usr_input = input("User: ")
    response_text = ""
    print("Assistant: ", end="", flush=True)
    for chunk in remote_chain.stream({"input": usr_input},
                                     {"configurable": {"conversation_id": conversation_id}}):
        response_text += chunk
        print(chunk, end="", flush=True)
    print()  # Ensure we move to the next line after the response is complete