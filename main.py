from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import time
import torch


# configs
model_file = 'better version/model/vietnamese-llama2-7b-40gb.Q2_K.gguf'
# model_file = 'model/Benchmaxx-Llama-3.2-1B-Instruct.IQ4_XS.gguf'
vector_db_path = 'vector_store/db_faiss'
model_name = "model/all-MiniLM-L6-v2-f16.gguf"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type='llama',
        max_new_tokens=256,
        temperature=0.01,
        use_fp16=torch.cuda.is_available()
    )
    return llm


# create prompt template
def create_prompt(template):
    # when using retrieval QA, always have to include the context (the text will be retrieved)
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    return prompt


# create simple chain
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        # k is the number of text that is closet to the query
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    return llm_chain


# read from vector db
def read_vector_db():
    embedding_model = GPT4AllEmbeddings(model_file=model_name)
    db = FAISS.load_local(vector_db_path,
                          embedding_model,
                          allow_dangerous_deserialization=True)
    return db


if __name__ == "__main__":
    print("loaded db")
    db = read_vector_db()
    print("loaded llm")
    llm = load_llm(model_file)

    template = """<|im_start|>system
    Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.
    {context}<|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    """

    prompt = create_prompt(template)
    print("chain created")
    llm_chain = create_qa_chain(prompt, llm, db)

    # run the chain
    start_time = time.time()
    # question = input("Hỏi gì liên quan tới research paper đi?\n")
    question = "Cách viết research paper"
    response = llm_chain.invoke({"query": question})
    print(response)
    print(time.time() - start_time)
