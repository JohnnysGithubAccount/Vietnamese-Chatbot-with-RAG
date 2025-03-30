import streamlit as st
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# Configuration
model_file = 'model/vinallama-7b-chat_q5_0.gguf'
vector_db_path = 'vector_store/db_faiss'
model_name = "model/all-MiniLM-L6-v2-f16.gguf"

# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type='llma',
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

# Create prompt template
def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variable=['context', 'question'])
    return prompt

# Create simple chain
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    return llm_chain

# Read from vector db
def read_vector_db():
    embedding_model = GPT4AllEmbeddings(model_file=model_name)
    db = FAISS.load_local(vector_db_path, embedding_model)
    return db


def create_db_from_file(uploaded_files):
    # Save uploaded files to a temporary directory
    temp_dir = 'temp_pdf_uploads'
    os.makedirs(temp_dir, exist_ok=True)

    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Load PDFs from the temporary directory
    loader = DirectoryLoader(temp_dir, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embedding_model = GPT4AllEmbeddings(model_file=model_name)
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

# Streamlit UI
st.title("Vietnamese Chatbot with RAG")
st.write("Ask me anything related to writing research papers!")

# File upload section
uploaded_files = st.file_uploader("Upload PDF Files", type='pdf', accept_multiple_files=True)

if st.button("Create Database"):
    if uploaded_files:
        db = create_db_from_file(uploaded_files)
        st.success("Database created successfully from uploaded PDF files.")
    else:
        st.warning("Please upload at least one PDF file.")

# Load the vector database and LLM
db = read_vector_db()
llm = load_llm(model_file)

# Chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Input question
question = st.text_input("You:", "")

if st.button("Send"):
    if question:
        st.session_state.history.append({"role": "user", "content": question})

        template = """Your prompt template here"""
        prompt = create_prompt(template)
        llm_chain = create_qa_chain(prompt, llm, db)

        response = llm_chain.invoke({"query": question})
        st.session_state.history.append({"role": "assistant", "content": response})

# Display chat history
if st.session_state.history:
    for chat in st.session_state.history:
        if chat['role'] == 'user':
            st.markdown(f"**You:** {chat['content']}")
        else:
            st.markdown(f"**Assistant:** {chat['content']}")