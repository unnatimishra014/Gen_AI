import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
import PyPDF2
import os
import hashlib
from transformers import pipeline

st.title("📄 Refund Policy Q&A")

uploaded_file = st.file_uploader("📂 Upload your company policy PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    full_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    st.write("✅ Text extracted from PDF!")

    file_bytes = uploaded_file.getvalue()
    doc_hash = hashlib.md5(file_bytes).hexdigest()
    index_path = f"faiss_index_{doc_hash}"

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(full_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(index_path):
        # Allow loading pickle safely because it's your own file
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        st.info("Loaded vector store from cache.")
    else:
        vector_store = FAISS.from_texts(texts, embeddings)
        vector_store.save_local(index_path)
        st.info("🗃️ Loaded cached vector store.\n🆕 Created new vector store and cached it.")

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})

    pipe = pipeline("text2text-generation", model="google/flan-t5-small")
    llm = HuggingFacePipeline(pipeline=pipe)

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    question = st.text_input("Ask a question about the uploaded document:")

    if question:
        with st.spinner("🔍 Finding answer..."):
            answer = qa.run(question)
        st.markdown("### Answer:")
        st.write(answer)
else:
    st.info("👉 Please upload a PDF file to get started.")
