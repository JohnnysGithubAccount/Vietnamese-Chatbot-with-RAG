import os
from re import template

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from litellm import max_tokens
from llama_cpp import Llama
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import CTransformers
from win32comext.adsi.demos.scp import verbose


def load_data(urls):
    # Load data
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()
    return data


def split_documents(data):
    # Text Splitter
    text_splitter = CharacterTextSplitter(
        separator='\n|。|\. ',
        chunk_size=800,
        chunk_overlap=160,
        length_function=len,
        is_separator_regex=True  # Bật regex cho separator
    )
    docs = text_splitter.split_documents(data)
    return docs


def initialize_chain(docs, temperature, k,
                     model_path: str = "model/EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf"):
    # Embeddings
    embedding_path = "model/all-MiniLM-L6-v2-f16.gguf"
    embeddings = GPT4AllEmbeddings(model_file=embedding_path)
    # embeddings = HuggingFaceEmbeddings()

    # Vector Store
    vectorStore = FAISS.from_documents(docs, embeddings)
    retriever = vectorStore.as_retriever(search_type="similarity",
                                         search_kwargs={"k": k})
    # Initialize LLM with llama.cpp
    # llm = Llama.from_pretrained(
    #     repo_id="bartowski/EXAONE-3.5-2.4B-Instruct-GGUF",
    #     filename="EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf",
    #     temperature=temperature,
    #     max_tokens=2000,  # Increased output length
    #     n_ctx=2048,       # Context window size
    #     n_gpu_layers=40,  # Number of layers to offload to GPU (if available)
    #     n_batch=512,      # Batch size for prompt processing
    #     verbose=True
    # )
    model_path = "model/vinallama-7b-chat_q5_0.gguf"

    llm = CTransformers(
        model=model_path,
        model_type='llama',
        max_new_tokens=512,
        temperature=temperature,
    )

    template_ = """<|im_start|>system
    Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.
    {context}<|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    """
    question_template = PromptTemplate(template=template_, input_variables=['context', 'question'])

    # qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,
    #                                            condense_question_prompt=QUESTION_PROMPT,
    #                                            return_source_documents=False, verbose=False)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": question_template},
        verbose=True
    )

    return qa

def conversational_chat(qa, query):
    try:
        result = qa.invoke({
            "query": query,
            # "chat_history": st.session_state.get('history', [])
        })
        answer = result["result"]
        st.session_state.setdefault('history', []).append((query, answer))
        return answer
    except Exception as e:
        st.error(f"Lỗi khi xử lý câu hỏi: {str(e)}")
        return "Xin lỗi, tôi gặp sự cố khi xử lý câu hỏi này."