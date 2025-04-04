import torch.cuda
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from accelerate import Accelerator
import gc

# configs
model_file = 'better version/model/vinallama-7b-chat_q5_0.gguf'
vector_db_path = 'vector_store/db_faiss'
model_name = "model/all-MiniLM-L6-v2-f16.gguf"


# load LLM
def load_llm(model_file):
    gpu_layers = 20

    config = {'max_new_tokens': 1024, 'temperature': 0.01,
              'gpu_layers': gpu_layers}

    llm = CTransformers(
        model=model_file,
        model_type='llma',
        max_new_tokens=1024,
        temperature=0.01,
        gpu_layers=gpu_layers,
        config=config
    )

    accelerator = Accelerator()
    llm, config = accelerator.prepare(llm, config)

    return llm


# create prompt template
def create_prompt(template):
    # when using retrieval QA, always have to include the context (the text will be retrieved)
    prompt = PromptTemplate(template=template, input_variable=['context', 'question'])

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    embedding_model = GPT4AllEmbeddings(model_file=model_name, device=device)
    db = FAISS.load_local(vector_db_path, embedding_model)

    return db


if __name__ == "__main__":
    print('reading database')
    db = read_vector_db()
    print('loading model')
    llm = load_llm(model_file)
    print('loaded successfully')
    template = """<|im_start|>system
    Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.
    {context}<|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    """

    prompt = create_prompt(template)

    llm_chain = create_qa_chain(prompt, llm, db)

    # run the chain
    # sample prompt: Phần Abstract trong một research paper có tác dụng gì?
    question = input("Bạn có thể hỏi tôi những thông tin liên quan tới cách viết research paper.\n")
    response = llm_chain.invoke({"query": question})
    print(response)
