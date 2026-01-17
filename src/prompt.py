# prompt_template = """
# Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Context: {context}
# Question: {question}
# Only return the helpful answer below nothing else.
# Helpful Answer:
# """ 

prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Use the following pieces of information to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""