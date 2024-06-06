# Create a button to trigger the clearing of cache and session states
if st.sidebar.button("Start a New Chat Interaction"):
    clear_cache_and_session()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
