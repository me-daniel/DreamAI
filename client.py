from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/")

while True:
    usr_input = input("User: ")
    print(remote_chain.invoke({"input": usr_input},
    {"configurable": {"session_id": 'unused'}}))