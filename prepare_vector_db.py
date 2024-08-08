from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
# read pdf file
from langchain_community.document_loaders import PyPDFLoader
# scan everything in a folder to find files for loading
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS

# define paths
data_path = "data"
vector_db_path = 'vector_store/db_faiss'


# create vector db from a simple text
def create_vector_db_from_text():
    raw_text = """
        Machine learning is a subfield of artificial intelligence that enables computers to learn from data and improve their performance over time without being explicitly programmed. 
        By using algorithms to identify patterns and make predictions, machine learning can automate complex tasks such as image recognition, natural language processing, and predictive analytics. 
        This technology is widely used in various industries, from healthcare for diagnosing diseases to finance for detecting fraudulent transactions. 
        As machine learning models are exposed to more data, they become more accurate and efficient, making them invaluable tools for solving real-world problems12.
        """

    # split the data
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)

    # embedding
    model_name = "model/all-MiniLM-L6-v2.gguf2.f16.gguf"
    embedding_model = GPT4AllEmbeddings(model_file=model_name)

    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    return db


def create_db_from_file(pdf_data_path: str = 'data'):
    # define loader to scan the data folder
    loader = DirectoryLoader(pdf_data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    model_name = "model/all-MiniLM-L6-v2.gguf2.f16.gguf"
    embedding_model = GPT4AllEmbeddings(model_file=model_name)

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db


if __name__ == "__main__":
    create_vector_db_from_text()
    create_db_from_file()
    