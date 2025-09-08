import os
import streamlit as st

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))


import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(openai_api_key=api_key)


# 1️⃣ Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2️⃣ Streamlit app title
st.title("📄 PDF Q&A App")

# 3️⃣ File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # 4️⃣ Read the PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # 5️⃣ Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # 6️⃣ Convert chunks into embeddings & store in FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # 7️⃣ Create a retriever and QA chain
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    # 8️⃣ Summarize the PDF
    st.subheader("📌 Summary of PDF")
    summary = llm.predict(f"Summarize this document in 5 bullet points:\n\n{text[:2000]}")
    st.write(summary)

    # 9️⃣ User question input
    st.subheader("Ask a Question from the PDF")
    query = st.text_input("Type your question here...")

    if query:
        result = qa_chain.run(query)
        if result:
            st.write("🤖 Answer:", result)
        else:
            st.write("❌ The requested data isn't available in this PDF.")

