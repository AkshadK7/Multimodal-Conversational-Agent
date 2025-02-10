from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

# Create necessary directories
if not os.path.exists('files'):
    os.mkdir('files')
if not os.path.exists('jj'):
    os.mkdir('jj')

# Set up session state
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question")

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(persist_directory='jj',
                                          embedding_function=OllamaEmbeddings(base_url='http://localhost:11434',
                                                                              model="mistral")
                                          )

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(base_url="http://localhost:11434",
                                  model="mistral",
                                  verbose=True,
                                  callback_manager=CallbackManager(
                                      [StreamingStdOutCallbackHandler()]),
                                  )

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Dropdown to select input type (CSV, PDF, Webpage)
option = st.selectbox("Select Input Type", ["CSV", "PDF", "Webpage"])

# Handle CSV input
if option == "CSV":
    st.title("CSV Conversational Agent")
    uploaded_file = st.file_uploader("Upload your CSV", type='csv')

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

    if uploaded_file is not None:
        file_path = "files/" + uploaded_file.name
        if not os.path.isfile(file_path):
            with st.status("Analyzing your document..."):
                bytes_data = uploaded_file.read()
                with open(file_path, "wb") as f:
                    f.write(bytes_data)

                # Load the CSV file into a DataFrame
                try:
                    df = pd.read_csv(file_path)
                    if df.empty:
                        st.error("The CSV file is empty!")
                        st.stop()
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
                    st.stop()

                # Convert DataFrame rows into LangChain Documents
                data = []
                for _, row in df.iterrows():
                    text_content = row.to_string(index=False)
                    data.append(Document(page_content=text_content))

                # Initialize text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len
                )
                all_splits = text_splitter.split_documents(data)

                # Create and persist the vector store
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=all_splits,
                    embedding=OllamaEmbeddings(model="mistral")
                )
                st.session_state.vectorstore.persist()

        st.session_state.retriever = st.session_state.vectorstore.as_retriever()

        # Initialize the QA chain
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type='stuff',
                retriever=st.session_state.retriever,
                verbose=True,
                chain_type_kwargs={
                    "verbose": True,
                    "prompt": st.session_state.prompt,
                    "memory": st.session_state.memory,
                }
            )

        # Chat input
        if user_input := st.chat_input("You:", key="user_input"):
            user_message = {"role": "user", "message": user_input}
            st.session_state.chat_history.append(user_message)
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Assistant is typing..."):
                    response = st.session_state.qa_chain(user_input)
                message_placeholder = st.empty()
                full_response = ""
                for chunk in response['result'].split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            chatbot_message = {"role": "assistant", "message": response['result']}
            st.session_state.chat_history.append(chatbot_message)

    else:
        st.write("Please upload a CSV file.")

# Handle Webpage input
elif option == "Webpage":
    st.title("Webpage Conversational Bot")
    url = st.text_input("Enter a webpage URL to scrape:")

    if url:
        try:
            with st.spinner("Scraping the webpage content..."):
                response = requests.get(url)
                response.raise_for_status()

                # Parse the webpage content
                soup = BeautifulSoup(response.text, 'html.parser')
                scraped_text = ' '.join([p.get_text() for p in soup.find_all('p')])

                # Initialize text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len
                )

                # Split the scraped text into chunks
                all_splits = text_splitter.create_documents([scraped_text])

                # Create and persist the vector store
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=all_splits,
                    embedding=OllamaEmbeddings(model="mistral")
                )
                st.session_state.vectorstore.persist()
                st.session_state.vectorstore_initialized = True

                st.success("Webpage content successfully scraped and processed!")

            st.session_state.retriever = st.session_state.vectorstore.as_retriever()

            # Initialize the QA chain
            if 'qa_chain' not in st.session_state:
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=st.session_state.llm,
                    chain_type='stuff',
                    retriever=st.session_state.retriever,
                    verbose=True,
                    chain_type_kwargs={
                        "verbose": True,
                        "prompt": st.session_state.prompt,
                        "memory": st.session_state.memory,
                    }
                )

            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["message"])

            # Chat input
            if user_input := st.chat_input("You:", key="user_input"):
                user_message = {"role": "user", "message": user_input}
                st.session_state.chat_history.append(user_message)
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    with st.spinner("Assistant is typing..."):
                        response = st.session_state.qa_chain(user_input)
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in response['result'].split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

                chatbot_message = {"role": "assistant", "message": response['result']}
                st.session_state.chat_history.append(chatbot_message)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.write("Please enter a URL to start.")

# Handle PDF input
elif option == "PDF":
    st.title("PDF Conversational Bot")
    uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

    if uploaded_file is not None:
        if not os.path.isfile("files/"+uploaded_file.name+".pdf"):
            with st.status("Analyzing your document..."):
                bytes_data = uploaded_file.read()
                f = open("files/"+uploaded_file.name+".pdf", "wb")
                f.write(bytes_data)
                f.close()
                loader = PyPDFLoader("files/"+uploaded_file.name+".pdf")
                data = loader.load()

                # Initialize text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len
                )
                all_splits = text_splitter.split_documents(data)

                # Create and persist the vector store
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=all_splits,
                    embedding=OllamaEmbeddings(model="mistral")
                )
                st.session_state.vectorstore.persist()

        st.session_state.retriever = st.session_state.vectorstore.as_retriever()

        # Initialize the QA chain
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type='stuff',
                retriever=st.session_state.retriever,
                verbose=True,
                chain_type_kwargs={
                    "verbose": True,
                    "prompt": st.session_state.prompt,
                    "memory": st.session_state.memory,
                }
            )

        # Chat input
        if user_input := st.chat_input("You:", key="user_input"):
            user_message = {"role": "user", "message": user_input}
            st.session_state.chat_history.append(user_message)
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Assistant is typing..."):
                    response = st.session_state.qa_chain(user_input)
                message_placeholder = st.empty()
                full_response = ""
                for chunk in response['result'].split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            chatbot_message = {"role": "assistant", "message": response['result']}
            st.session_state.chat_history.append(chatbot_message)

    else:
        st.write("Please upload a PDF file.")
