import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store


def get_conversation_chain(vector_store):
   
    # prompt_template = """ Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    #     provided context just say, "Answer is not available in the context.". Do not provide the wrong answer\n\n
        
    #     Context:\n {context}?\n
    #     Question: \n{question}\n

    #     Answer:
    # """
    # prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    llm = llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,  retriever=vector_store.as_retriever(),
        memory=memory)
    
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    # st.write(response)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.chat_message("user").markdown(message.content)
        else:
            st.chat_message("assistant").markdown(message.content)


def main():
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Multiple PDFs :books:")

    user_question = st.chat_input("Ask anything about your uploaded documents")

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader('Upload Documents')
        pdf_docs = st.file_uploader("Upload your files here and click on Process", accept_multiple_files=True)
        
        if st.button('Process'):
            with st.spinner("Processing..."):
                # get text from pdfs
                raw_text = get_pdf_text(pdf_docs)

                # break text into chunks
                text_chunks = get_text_chunks(raw_text)
                
                # store text chunks in vector db
                vector_store = get_vector_store(text_chunks)

                # get conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)

                st.success('Done')



if __name__ == '__main__':
    main()