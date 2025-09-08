import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Load API key ---
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not api_key:
    st.error("❌ No OpenAI API key found. Please set it in Streamlit Secrets or .env file.")
    st.stop()

# --- Streamlit UI ---
st.title("📄 PDF Q&A App")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # 1. Read PDF
    pdf_reader = PdfReader(uploaded_file)
    text = "".join([page.extract_text() for page in pdf_reader.pages])

    # 2. Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # 3. Build embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # 4. Store in FAISS
    vectorstore = FAISS.from_texts(chunks, embeddings)

    st.success("✅ PDF processed. You can now ask questions!")


# Load API key safely
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not api_key:
    st.error("❌ No OpenAI API key found. Please add it in Streamlit Secrets or .env")
    st.stop()

# Pass key explicitly
embeddings = OpenAIEmbeddings(openai_api_key=api_key)


from dotenv import load_dotenv
load_dotenv()


# 6️⃣ User question input
question = st.text_input("Ask a question about the PDF:")

if question:
    # Search in FAISS (retrieves most relevant chunks)
    docs = vectorstore.similarity_search(question, k=3)

    # Build a prompt for the LLM
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""
    You are an assistant. Use ONLY the context below to answer.
    If the answer is not in the context, say:
    'The requested data isn't available in this PDF.'

    Context:
    {context}

    Question: {question}
    """

    # Get response
    answer = llm.predict(prompt)

    # Show result
    st.markdown("### 📖 Answer")
    st.write(answer)



