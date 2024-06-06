def chat_ui(qa):
    # Accept user input
    if prompt := st.chat_input(
        "Ask me questions: How can I retrieve data from Deep Lake in Langchain?"
    ):

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Load the memory variables, which include the chat history
            memory_variables = memory.load_memory_variables({})

            # Predict the AI's response in the conversation
            with st.spinner("Searching course material"):
                response = capture_and_display_output(
                    qa, ({"question": prompt, "chat_history": memory_variables})
                )

            # Display chat response
            full_response += response["answer"]
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            #Display top 2 retrieved sources
            source = response["source_documents"][0].metadata
            source2 = response["source_documents"][1].metadata
            with st.expander("See Resources"):
                st.write(f"Title: {source['title'].split('·')[0].strip()}")
                st.write(f"Source: {source['source']}")
                st.write(f"Relevance to Query: {source['relevance_score'] * 100}%")
                st.write(f"Title: {source2['title'].split('·')[0].strip()}")
                st.write(f"Source: {source2['source']}")
                st.write(f"Relevance to Query: {source2['relevance_score'] * 100}%")

        # Append message to session state
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

# Run function passing the ConversationalRetrievalChain
chat_ui(qa)
