from langchain_community.llms import CTransformers

# Use forward slashes for path (works on Windows too)
model_path = "model/vinallama-7b-chat_q5_0.gguf"

llm = CTransformers(
    model=model_path,
    model_type='llama',
    max_new_tokens=256,
    temperature=0.01,
)


# Make a prediction
prompt = "Giải thích khái niệm machine learning bằng tiếng Việt"  # Vietnamese prompt
response = llm(prompt)

print("Generated response:")
print(response)