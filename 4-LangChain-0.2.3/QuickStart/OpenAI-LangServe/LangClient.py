from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/agent/")
result = remote_chain.invoke({
    "input": "how can langsmith help with testing?",
    "chat_history": []  # Providing an empty list as this is the first call
})
print(result.get("output"))