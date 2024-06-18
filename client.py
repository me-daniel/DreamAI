from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
#remote_chain = RemoteRunnable("http://localhost:8000/chain_with_history/")
while True:
    usr_input = input("User: ")
    print(remote_chain.invoke({"input": usr_input},
    {"configurable": {"session_id": 'unused'}}))