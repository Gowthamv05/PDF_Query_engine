# app.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------------------
# 1️⃣ Get OpenAI API key from Streamlit secrets
# ----------------------------
api_key = st.secrets["OPENAI_API_KEY"]

# ----------------------------
# 2️⃣ Initialize LLM and embeddings
# ----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# ----------------------------
# 3️⃣ Streamlit UI
# ----------------------------
st.title("📄 PDF Q&A App")
st.write("Upload a PDF, then ask questions about its content!")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # ----------------------------
    # 4️⃣ Read PDF content
    # ----------------------------
    pdf_reader = PdfReader(uploaded_file)
    text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

    if not text:
        st.error("❌ Could not extract text from this PDF.")
        st.stop()

    # ----------------------------
    # 5️⃣ Split text into chunks
    # ----------------------------
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # ----------------------------
    # 6️⃣ Build FAISS vectorstore (pass embeddings with API key!)
    # ----------------------------
    vectorstore = FAISS.from_texts(chunks, embeddings)

    st.success("✅ PDF processed. You can now ask questions!")

    # ----------------------------
    # 7️⃣ User asks a question
    # ----------------------------
    question = st.text_input("Ask a question about the PDF:")

    if question:
        # Retrieve top 3 similar chunks
        docs = vectorstore.similarity_search(question, k=3)

        # Build context for LLM
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""
        You are an assistant. Use ONLY the context below to answer.
        If the answer is not in the context, say:
        'The requested data isn't available in this PDF.'

        Context:
        {context}

        Question: {question}
        """

        # Generate answer
        answer = llm.predict(prompt)

        # Show result
        st.markdown("### 📖 Answer")
        st.write(answer)
