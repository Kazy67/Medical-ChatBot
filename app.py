from flask import Flask, render_template, request
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma
from src.helper import download_huggingface_embeddings
from src.prompt import *

app = Flask(__name__)

# 1. LOAD existing embeddings and database (Fast!)
embeddings = download_huggingface_embeddings()
docsearch = Chroma(persist_directory="db", embedding_function=embeddings)

# 2. Setup Prompt and Llama 3
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = LlamaCpp(
    model_path="model/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=12,
    temperature=0.3, # Lowering temperature helps with medical accuracy
    stop=["<|eot_id|>", "<|begin_of_text|>"], # Crucial to prevent prompt leakage
    verbose=False
)

# 3. Create the Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"]) # Add GET back just for testing
def chat():
    # Use request.form if it's a POST, or request.args if it's a GET
    msg = request.form.get("msg") or request.args.get("msg")
    
    if not msg:
        return "Error: No message received"

    print(f"User Message: {msg}") # This will show in your terminal
    result = qa.invoke({"query": msg})
    
    # Debug: Print the result to terminal to see if Llama 3 is thinking
    print(f"Llama 3 Response: {result['result']}")
    
    return str(result["result"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)